# ------------------------------------------------------------------------
# reid_query_module.py
#
# Trajectory ReID module following the paper's architecture (see figure):
#
#   Queue {t-tau2, ..., t-1, t}
#       -> Frame Embedding (sinusoidal, eqs. 5 & 6)
#       -> Transformer Layers (L_R blocks from build_transformer / 
#          build_transformer_local — the provided ReID ViT backbone,
#          bypassing patch embedding, feeding track embeddings directly)
#       -> F_id
#       -> Output Layer (BN + classifier head, ID loss)
#
# Key design: we skip the ViT patch embedding (which expects image pixels)
# and feed the track embedding sequence directly into self.base.blocks,
# after projecting from track_dim -> vit_dim with a linear layer.
# The CLS token output of the last block is taken as F_id.
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import importlib.util
from importlib.machinery import SourceFileLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.structures import Instances
from .temp import QueueMemoryBank, TrackQueue, TrackSnapshot


def _load_transreid_vit_module():
    """Load TransReID ViT definitions directly from models/structures/vit_pytorch."""
    try:
        from .structures import vit_pytorch as vit_module
        return vit_module
    except Exception:
        vit_path = Path(__file__).resolve().parent / 'structures' / 'vit_pytorch'
        if not vit_path.exists():
            raise ImportError(f'Cannot locate TransReID ViT file at: {vit_path}')

        loader = SourceFileLoader('motr_reid_query_vit', str(vit_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None:
            raise ImportError(f'Cannot create import spec for: {vit_path}')
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module


class ReIDViTModel(nn.Module):
    """Adapter that exposes the interface expected by TrajectoryReIDBackbone."""

    def __init__(self, base: nn.Module, in_planes: int):
        super().__init__()
        self.base = base
        self.in_planes = int(in_planes)


def build_reid_query_vit_model(
    transformer_type: str = 'vit_base_patch16_224_TransReID',
    img_size: Tuple[int, int] = (256, 128),
    stride_size: int = 16,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    camera_num: int = 0,
    view_num: int = 0,
    sie_xishu: float = 1.0,
) -> ReIDViTModel:
    """Build ReIDQuery backbone model directly from vit_pytorch definitions."""
    vit_module = _load_transreid_vit_module()
    factory = {
        'vit_base_patch16_224_TransReID': vit_module.vit_base_patch16_224_TransReID,
        'deit_base_patch16_224_TransReID': vit_module.vit_base_patch16_224_TransReID,
        'vit_small_patch16_224_TransReID': vit_module.vit_small_patch16_224_TransReID,
        'deit_small_patch16_224_TransReID': vit_module.deit_small_patch16_224_TransReID,
    }
    in_planes_map = {
        'vit_base_patch16_224_TransReID': 768,
        'deit_base_patch16_224_TransReID': 768,
        'vit_small_patch16_224_TransReID': 768,
        'deit_small_patch16_224_TransReID': 384,
    }

    if transformer_type not in factory:
        supported = ', '.join(sorted(factory.keys()))
        raise ValueError(f'Unsupported transformer_type={transformer_type!r}. Supported: {supported}')

    base = factory[transformer_type](
        img_size=img_size,
        stride_size=stride_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        camera=camera_num,
        view=view_num,
        local_feature=False,
        sie_xishu=sie_xishu,
    )
    return ReIDViTModel(base=base, in_planes=in_planes_map[transformer_type])


# ---------------------------------------------------------------------------
# 1. Frame Embedding  (equations 5 & 6 from the paper)
# ---------------------------------------------------------------------------

class FrameEmbedding(nn.Module):
    """
    Sinusoidal encoding of globally-unique frame indices.

        FE_(frame, 2i)   = sin( frame / 10000^(2i/d) )     [eq. 5]
        FE_(frame, 2i+1) = cos( frame / 10000^(2i/d) )     [eq. 6]

    Added element-wise to the track embedding tokens before the
    transformer layers to inject temporal ordering information.

    Input  : frame_indices [L]  long tensor of globally unique frame ids
    Output : fe            [L, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, frame_indices: Tensor) -> Tensor:
        device = frame_indices.device
        d      = self.d_model

        dim_idx = torch.arange(d, dtype=torch.float32, device=device)
        exp     = (2 * (dim_idx // 2)) / d
        denom   = torch.pow(torch.tensor(10000.0, device=device), exp)   # [d]

        angle = frame_indices.float().unsqueeze(1) / denom.unsqueeze(0)  # [L, d]

        fe = torch.zeros_like(angle)
        fe[:, 0::2] = torch.sin(angle[:, 0::2])   # even dims -> sin
        fe[:, 1::2] = torch.cos(angle[:, 1::2])   # odd  dims -> cos

        return fe   # [L, d_model]


# ---------------------------------------------------------------------------
# 2. ReID Transformer Backbone wrapper
#
#    Wraps the provided build_transformer / build_transformer_local so
#    we can feed track embedding sequences instead of image pixels.
#
#    The ViT in build_transformer has this structure:
#        patch_embed  (image -> patch tokens)  <- we SKIP this
#        cls_token                              <- we PREPEND this
#        pos_embed                              <- we SKIP (use Frame Embedding instead)
#        blocks[0..L-1]                         <- we USE these (the L_R layers)
#        norm                                   <- we USE this
#
#    Our input path:
#        track_emb [N_alive, tau2, track_dim]
#            -> input_proj  [N_alive, tau2, vit_dim]
#            -> prepend cls_token -> [N_alive, tau2+1, vit_dim]
#            -> base.blocks (all L_R transformer layers)
#            -> base.norm
#            -> cls output [:, 0, :] -> F_id [N_alive, vit_dim]
# ---------------------------------------------------------------------------

class TrajectoryReIDBackbone(nn.Module):
    """
    Wraps the ViT backbone from build_transformer to process sequences
    of track embeddings (not image patches).

    Args:
        reid_model  : instance of build_transformer or build_transformer_local
                      (already built and optionally pretrained)
        track_dim   : dimension of track_instances.output_embedding
                      (= transformer.d_model // 2 in MOTR)

    The vit_dim is read from reid_model.in_planes (set by build_transformer).
    """

    def __init__(self, reid_model: nn.Module, track_dim: int):
        super().__init__()

        # Store a reference to the ViT base from build_transformer
        # reid_model.base is the actual ViT (vit_base_patch16_224_TransReID etc.)
        self.vit = reid_model.base

        vit_dim = reid_model.in_planes   # 768 for vit_base, 384 for deit_small

        # Project track_dim -> vit_dim so our embeddings fit the ViT blocks
        self.input_proj = nn.Linear(track_dim, vit_dim)

        # Learnable CLS token (same role as in the original ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.vit_dim = vit_dim

    def forward(
        self,
        seq      : Tensor,                          # [N_alive, tau2, track_dim]
        pad_mask : Tensor,                          # [N_alive, tau2]  True=pad
        cam_label: Optional[Tensor] = None,         # [N_alive] long, for SIE
        view_label: Optional[Tensor] = None,        # [N_alive] long, for SIE
    ) -> Tensor:
        """
        Returns:
            F_id : [N_alive, vit_dim]  — CLS token output of the last ViT block
        """
        N = seq.shape[0]

        # ----------------------------------------------------------
        # Project track embeddings into ViT dimension
        # seq: [N_alive, tau2, track_dim] -> [N_alive, tau2, vit_dim]
        # ----------------------------------------------------------
        x = self.input_proj(seq)                    # [N_alive, tau2, vit_dim]

        # ----------------------------------------------------------
        # Optionally add SIE (Side Information Embedding) if the ViT
        # was trained with camera/view conditioning.
        # SIE is a learned embedding in self.vit.sie_embed indexed by
        # cam_label/view_label — added to the CLS token only.
        # ----------------------------------------------------------
        cls = self.cls_token.expand(N, -1, -1)      # [N_alive, 1, vit_dim]

        if cam_label is not None and hasattr(self.vit, 'sie_embed'):
            # self.vit.sie_xishu scales the SIE contribution
            sie = self.vit.sie_embed[cam_label].unsqueeze(1)   # [N_alive, 1, vit_dim]
            cls = cls + self.vit.sie_xishu * sie

        if view_label is not None and hasattr(self.vit, 'sie_embed'):
            sie = self.vit.sie_embed[view_label].unsqueeze(1)
            cls = cls + self.vit.sie_xishu * sie

        # ----------------------------------------------------------
        # Prepend CLS token -> [N_alive, tau2+1, vit_dim]
        # Extend pad_mask  -> [N_alive, tau2+1]  (CLS is never masked)
        # ----------------------------------------------------------
        x = torch.cat([cls, x], dim=1)              # [N_alive, tau2+1, vit_dim]

        cls_mask = torch.zeros(N, 1, dtype=torch.bool, device=seq.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)  # [N_alive, tau2+1]

        # ----------------------------------------------------------
        # Forward through ALL ViT transformer blocks (the L_R layers).
        # self.vit.blocks is nn.ModuleList of Block objects.
        # Each Block is a standard ViT block:
        #   norm1 -> MHSA -> residual -> norm2 -> MLP -> residual
        #
        # PyTorch ViT blocks accept (x,) without explicit mask by default,
        # so we handle padding via zeroing out padded positions after each block.
        # ----------------------------------------------------------
        for block in self.vit.blocks:
            x = block(x)
            # Zero out padded positions so they don't influence subsequent
            # blocks through residual connections
            x[full_mask] = 0.0

        x = self.vit.norm(x)                        # [N_alive, tau2+1, vit_dim]

        # CLS token = fused trajectory representation
        F_id = x[:, 0]                              # [N_alive, vit_dim]
        return F_id


# ---------------------------------------------------------------------------
# 3. Output Layer: BN + classifier head + ID loss
#    CE loss  : cross-entropy over tracklet identity classes
#    Triplet  : batch-hard triplet loss for metric learning
#    L_R = L_CE + L_triplet
# ---------------------------------------------------------------------------

def _batch_hard_triplet_loss(
    embeddings: Tensor,          # [N, D]  L2-normalised features
    labels    : Tensor,          # [N]     long identity labels
    margin    : float = 0.3,
) -> Tensor:
    """
    Batch-hard triplet loss (Hermans et al., 2017).

    For each anchor i:
        hardest positive  = same identity, maximum distance
        hardest negative  = different identity, minimum distance

        loss_i = max(0,  d(a, p+)  -  d(a, n-)  +  margin )

    Uses squared Euclidean distance on L2-normalised embeddings
    (equivalent to 2 - 2*cosine_similarity, numerically stable).

    Args:
        embeddings : [N, D]  already L2-normalised
        labels     : [N]     identity label per sample
        margin     : scalar  triplet margin (default 0.3, same as TransReID)

    Returns:
        loss : scalar mean over all valid anchors
    """
    # Pairwise squared distance matrix  [N, N]
    # ||a - b||^2 = 2 - 2*(a . b)  for unit vectors
    dot   = torch.matmul(embeddings, embeddings.t())   # [N, N]
    dist  = (2.0 - 2.0 * dot).clamp(min=0.0)          # numerical safety

    N = labels.shape[0]
    # Boolean masks  [N, N]
    same_id = labels.unsqueeze(1) == labels.unsqueeze(0)   # True where same identity
    diff_id = ~same_id

    # Mask out diagonal (self-pairs)
    eye = torch.eye(N, dtype=torch.bool, device=embeddings.device)
    same_id = same_id & ~eye

    # Hardest positive: largest distance among same-identity pairs
    # Replace invalid entries with -inf before max
    dist_ap = (dist * same_id.float()).masked_fill(~same_id, -1e9)
    hardest_pos, _ = dist_ap.max(dim=1)   # [N]

    # Hardest negative: smallest distance among different-identity pairs
    dist_an = (dist * diff_id.float()).masked_fill(~diff_id,  1e9)
    hardest_neg, _ = dist_an.min(dim=1)   # [N]

    # Triplet loss with soft margin (clamped at 0)
    loss_per_anchor = F.relu(hardest_pos - hardest_neg + margin)

    # Only average over anchors that have at least one valid positive
    valid = same_id.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss_per_anchor[valid].mean()


class ReIDOutputLayer(nn.Module):
    """
    Output Layer for Trajectory ReID:
        BN normalisation -> CE loss + Batch-Hard Triplet loss

        L_R = L_CE + L_triplet

    Mirrors the bottleneck + classifier structure of build_transformer
    but adds triplet loss on the pre-BN (global) features, matching
    the standard TransReID / FairMOT training recipe.

    Training : returns (BN-feat [N, D],  {'ce': ..., 'triplet': ..., 'total': ...})
    Inference: returns (BN-feat [N, D],  None)
    """

    def __init__(
        self,
        vit_dim        : int,
        num_ids        : int,
        label_smoothing: float = 0.1,
        triplet_margin : float = 0.3,
    ):
        super().__init__()
        self.bottleneck     = nn.BatchNorm1d(vit_dim)
        self.classifier     = nn.Linear(vit_dim, num_ids, bias=False)
        self.label_smoothing = label_smoothing
        self.triplet_margin  = triplet_margin

        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias,   0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(
        self,
        F_id      : Tensor,                        # [N, vit_dim]  raw CLS output
        target_ids: Optional[Tensor] = None,       # [N] long identity labels
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Returns:
            feat      : [N, vit_dim]  BN-normalised feature (for downstream use)
            loss_dict : {
                          'ce'     : CE loss scalar,
                          'triplet': triplet loss scalar,
                          'total'  : L_CE + L_triplet  (= L_R in eq. 7)
                        }
                        or None at inference.
        """
        # BN-normalised feature — used for CE loss and downstream embedding.
        # BatchNorm1d in training mode requires N > 1. With many camera views,
        # some frames can produce a singleton alive-track batch (N == 1).
        # In that case, fall back to running-stat normalization to avoid crash.
        if self.training and F_id.shape[0] <= 1:
            feat = F.batch_norm(
                F_id,
                self.bottleneck.running_mean,
                self.bottleneck.running_var,
                self.bottleneck.weight,
                self.bottleneck.bias,
                training=False,
                momentum=self.bottleneck.momentum,
                eps=self.bottleneck.eps,
            )
        else:
            feat = self.bottleneck(F_id)           # [N, vit_dim]

        if self.training and target_ids is not None:
            # --- CE loss (classification) ---
            logits   = self.classifier(feat)       # [N, num_ids]
            loss_ce  = F.cross_entropy(
                logits, target_ids,
                label_smoothing=self.label_smoothing,
            )

            # --- Triplet loss (metric learning) ---
            # Triplet is computed on L2-normalised pre-BN features (F_id),
            # following standard TransReID practice: BN distorts distances,
            # so raw features are better for triplet.
            normed = F.normalize(F_id, p=2, dim=1) # [N, vit_dim]
            loss_triplet = _batch_hard_triplet_loss(
                normed, target_ids, self.triplet_margin
            )

            loss_reid = loss_ce + loss_triplet

            return feat, {
                'ce'     : loss_ce,
                'triplet': loss_triplet,
                'total'  : loss_reid,              # L_R for UncertaintyWeightedLoss
            }

        return feat, None


# ---------------------------------------------------------------------------
# 4. ReIDQueryModule  -- main class
# ---------------------------------------------------------------------------

class ReIDQueryModule(nn.Module):
    """
    Full Trajectory ReID module following the paper figure:

        Queue snapshots {t-tau2, ..., t}
            -> Frame Embedding  (sinusoidal, eqs. 5 & 6)
            -> ViT blocks from build_transformer  (the L_R transformer layers)
            -> F_id  [N_alive, vit_dim]
            -> Output Layer  (BN + classifier, ID loss)
            -> track_instances.output_embedding updated

    Args:
        reid_model      : build_transformer or build_transformer_local instance
                          (the provided ReID model with pretrained ViT weights)
        track_dim       : dim of track_instances.output_embedding
                          (= transformer.d_model // 2 in MOTR)
        num_ids         : number of tracklet identity classes
        tau1            : maximum queue depth (= QueueMemoryBank.maxlen)
        tau2            : window length fed to the transformer  (tau2 <= tau1)
        label_smoothing : for cross-entropy ID loss
        temporal_decay_alpha : if <1.0, apply exponential decay to older tokens in the window -> older frames contribute less when feed into the encoder.

    Integration in MultiviewMOTR._post_process_single_image,
    after assigning output_embedding and BEFORE match_for_single_frame:

        track_instances.output_embedding = frame_res['hs'][0]

        track_instances, id_loss = self.reid_module(
            track_instances  = track_instances,
            queue_bank       = self.memory_bank,       # QueueMemoryBank
            cam_idx          = cam_idx,
            global_frame_idx = global_frame_idx,
            target_ids       = gt_track_ids,           # None at inference
        )
        if id_loss is not None:
            self.criterion.get_criterion(cam_idx)\\
                .losses_dict[f'frame_{frame_idx}_reid_loss'] = id_loss
    """

    def __init__(
        self,
        reid_model     : nn.Module,
        track_dim      : int,
        num_ids        : int,
        tau1           : int   = 10,
        tau2           : int   = 4,
        temporal_decay_alpha: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        assert tau2 <= tau1, f"tau2 ({tau2}) must be <= tau1 ({tau1})"
        if temporal_decay_alpha < 0.0 or temporal_decay_alpha > 1.0:
            raise ValueError(
                f"temporal_decay_alpha must be in [0, 1], got {temporal_decay_alpha}"
            )
        self.tau2      = tau2
        self.track_dim = track_dim
        self.temporal_decay_alpha = float(temporal_decay_alpha)

        # Frame Embedding: uses track_dim because it is added BEFORE projection
        # into vit_dim (we want FE to encode position in the track space)
        self.frame_embedding = FrameEmbedding(track_dim)

        # Transformer backbone: wraps ViT blocks from the provided reid_model
        self.backbone = TrajectoryReIDBackbone(reid_model, track_dim)

        # Output layer
        self.output_layer = ReIDOutputLayer(
            vit_dim         = self.backbone.vit_dim,
            num_ids         = num_ids,
            label_smoothing = label_smoothing,
        )

    # ------------------------------------------------------------------
    # Internal: extract tau2-length sequence from one TrackQueue
    # ------------------------------------------------------------------

    def _build_sequence(
        self,
        track_queue: TrackQueue,
        device     : torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Reads the tau2 most-recent TrackSnapshots from a TrackQueue and
        returns left-padded tensors ready for the transformer encoder.

        Edge cases
        ----------
        T_actual == 0
            Safety net only. forward() already catches get_queue()==None
            and len(queue)==0 before ever calling here. If somehow reached,
            returns all-zeros + all-True mask (fully padded, no valid tokens).

        T_actual == 1  (track born this frame / first detection in clip)
            The queue has exactly 1 snapshot (just pushed by forward()).
            Single valid token placed at position [-1]; positions [0..tau2-2]
            are zero-padded and masked True. The ViT attends to 1 token only.

        T_actual in [2, tau2-1]  (track alive but window not yet full)
            Can happen in two situations:
              a) Track is younger than tau2 frames.
              b) Track dipped below score_thresh for some frames -- those
                 frames were never pushed into the queue, creating a gap.
            Valid snapshots are right-aligned; left positions are padded.

        T_actual == tau2  (full window, normal steady-state operation)
            No padding. All tau2 positions are valid.

        Returns
        -------
        frame_indices    : [tau2]              long   (0 for padded slots)
        embeddings       : [tau2, track_dim]   float  (zeros for padded slots)
        key_padding_mask : [tau2]              bool   True = padded / ignore
        """
        snapshots: List[TrackSnapshot] = list(track_queue.snapshots)[-self.tau2:]
        T_actual  = len(snapshots)

        # Allocate fully-padded output tensors.
        frame_indices    = torch.zeros(self.tau2, dtype=torch.long,  device=device)
        embeddings       = torch.zeros(self.tau2, self.track_dim,    device=device)
        key_padding_mask = torch.ones( self.tau2, dtype=torch.bool,  device=device)

        if T_actual == 0:
            # Safety net — return all-padded (no valid tokens).
            return frame_indices, embeddings, key_padding_mask

        # Left-pad: valid snapshots occupy the rightmost T_actual positions.
        #
        #   index : [ 0  ...  pad_offset-1 | pad_offset  ...  tau2-1 ]
        #   data  : [ 0 (pad) ... 0 (pad)  | snap[0]  ...  snap[-1]  ]
        #   mask  : [ True    ... True      | False    ...  False     ]
        pad_offset = self.tau2 - T_actual

        fidxs = torch.tensor(
            [s.frame_id for s in snapshots],
            dtype=torch.long, device=device,
        )                                                        # [T_actual]

        # QueueMemoryBank stores embeddings on CPU via detach().cpu() in push()
        # -- move to the target device before stacking into the batch tensor.
        embeds = torch.stack(
            [s.embedding for s in snapshots], dim=0
        ).to(device)                                             # [T_actual, track_dim]

        frame_indices   [pad_offset:] = fidxs
        embeddings      [pad_offset:] = embeds
        key_padding_mask[pad_offset:] = False                   # mark as valid

        if self.temporal_decay_alpha < 1.0:
            valid_count = T_actual
            if valid_count > 1:
                decay_base = torch.tensor(
                    self.temporal_decay_alpha,
                    dtype=embeddings.dtype,
                    device=device,
                )
                # Oldest valid token gets alpha^(valid_count-1), newest gets alpha^0.
                exponents = torch.arange(
                    valid_count - 1,
                    -1,
                    -1,
                    dtype=embeddings.dtype,
                    device=device,
                )
                decay = torch.pow(decay_base, exponents).unsqueeze(-1)
                embeddings[pad_offset:] = embeddings[pad_offset:] * decay

        return frame_indices, embeddings, key_padding_mask


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        track_instances : Instances,
        queue_bank      : QueueMemoryBank,
        cam_idx         : int,
        global_frame_idx: int,
        target_ids      : Optional[Tensor] = None,
        cam_label       : Optional[int]    = None,
        view_label      : Optional[int]    = None,
    ) -> Tuple[Instances, Optional[Tensor]]:
        """
        Args:
            track_instances  : output of _forward_single_image.
                               Requires: .obj_idxes, .output_embedding,
                               .scores, .query_pos.
            queue_bank       : shared QueueMemoryBank (maxlen must = tau1).
            cam_idx          : camera index for this frame.
            global_frame_idx : globally unique frame index, same value for
                               all cameras at the same timestamp.
            target_ids       : [N_alive] long ground-truth IDs. None at inference.
            cam_label/view_label : SIE conditioning for the ViT (optional).

        Returns:
            track_instances  : .output_embedding replaced with ReID F_id
                               for all alive tracks.
            id_loss          : scalar CE loss (training) or None (inference).
        """
        device = track_instances.output_embedding.device

        # ----------------------------------------------------------------
        # Step 1 -- push current frame's embeddings into QueueMemoryBank.
        #   push() applies score_thresh filtering and stores
        #   TrackSnapshot(embedding, query_pos, score, frame_id).
        #   Push FIRST so the current frame is in the window.
        # ----------------------------------------------------------------
        queue_bank.push(
            cam_idx         = cam_idx,
            track_instances = track_instances,
            frame_id        = global_frame_idx,
        )

        # ----------------------------------------------------------------
        # Step 2 -- find alive tracks
        # ----------------------------------------------------------------
        alive_mask    = track_instances.obj_idxes >= 0
        alive_indices = alive_mask.nonzero(as_tuple=False).squeeze(1)  # [N_alive]
        N_alive       = len(alive_indices)

        if N_alive == 0:
            return track_instances, None

        alive_obj_ids = track_instances.obj_idxes[alive_indices].tolist()

        # ----------------------------------------------------------------
        # Step 3 -- build batched sequence tensors from the queue.
        #
        #   For each alive track:
        #     queue_bank.get_queue(cam_idx, obj_id) -> TrackQueue
        #     _build_sequence() -> left-padded [tau2, track_dim] + mask
        # ----------------------------------------------------------------
        seq_list, fidx_list, pad_list = [], [], []

        for obj_id in alive_obj_ids:
            track_queue = queue_bank.get_queue(cam_idx, int(obj_id))

            if track_queue is None or len(track_queue) == 0:
                # New track or below score_thresh -- all-zero fallback
                fidxs    = torch.zeros(self.tau2, dtype=torch.long,  device=device)
                embeds   = torch.zeros(self.tau2, self.track_dim,    device=device)
                pad_mask = torch.ones( self.tau2, dtype=torch.bool,  device=device)
            else:
                fidxs, embeds, pad_mask = self._build_sequence(track_queue, device)

            fidx_list.append(fidxs)
            seq_list.append(embeds)
            pad_list.append(pad_mask)

        seq_batch  = torch.stack(seq_list,  dim=0)   # [N_alive, tau2, track_dim]
        fidx_batch = torch.stack(fidx_list, dim=0)   # [N_alive, tau2]
        pad_batch  = torch.stack(pad_list,  dim=0)   # [N_alive, tau2]

        # ----------------------------------------------------------------
        # Step 4 -- add Frame Embedding element-wise (in track_dim space,
        #   before projection into vit_dim).
        #
        #   Computed per-track because gaps from missed detections mean
        #   each track can have different frame indices in its window.
        # ----------------------------------------------------------------
        fe_batch = torch.stack(
            [self.frame_embedding(fidx_batch[i]) for i in range(N_alive)],
            dim=0,
        )                                             # [N_alive, tau2, track_dim]
        seq_batch = seq_batch + fe_batch             # element-wise add

        # ----------------------------------------------------------------
        # Step 5 -- forward through ViT transformer blocks (L_R layers).
        #
        #   TrajectoryReIDBackbone:
        #     input_proj  : [N_alive, tau2, track_dim] -> [N_alive, tau2, vit_dim]
        #     prepend CLS : [N_alive, tau2+1, vit_dim]
        #     vit.blocks  : all L_R transformer attention layers
        #     vit.norm
        #     cls output  : [N_alive, vit_dim]  -> F_id
        # ----------------------------------------------------------------
        cam_t  = (torch.full((N_alive,), cam_label,  dtype=torch.long, device=device)
                  if cam_label  is not None else None)
        view_t = (torch.full((N_alive,), view_label, dtype=torch.long, device=device)
                  if view_label is not None else None)

        F_id = self.backbone(seq_batch, pad_batch, cam_t, view_t)   # [N_alive, vit_dim]

        # ----------------------------------------------------------------
        # Step 6 -- Output Layer: BN normalisation + CE + Triplet losses
        #
        #   output_layer returns:
        #     feat      : [N_alive, vit_dim]  BN-normalised feature
        #     loss_dict : {'ce': ..., 'triplet': ..., 'total': ...}
        #                 or None at inference
        # ----------------------------------------------------------------
        feat, loss_dict = self.output_layer(F_id, target_ids)      # [N_alive, vit_dim]

        # ----------------------------------------------------------------
        # Step 7 -- write F_id back into track_instances.output_embedding.
        #
        #   If vit_dim != track_dim, project back so the downstream QIM
        #   receives the same dimension it expects.
        #   Only alive tracks are overwritten; new/unmatched slots keep
        #   their raw transformer embedding unchanged.
        # ----------------------------------------------------------------
        updated = track_instances.output_embedding.clone()

        if self.backbone.vit_dim != self.track_dim:
            if not hasattr(self, 'output_proj'):
                self.output_proj = nn.Linear(
                    self.backbone.vit_dim, self.track_dim
                ).to(device)
            feat = self.output_proj(feat)                           # [N_alive, track_dim]

        updated[alive_indices] = feat
        track_instances.output_embedding = updated

        return track_instances, loss_dict
        # loss_dict is None at inference.
        # At training, pass loss_dict['total'] as L_R to UncertaintyWeightedLoss,
        # and log loss_dict['ce'] / loss_dict['triplet'] separately.

# ---------------------------------------------------------------------------
# 5. Uncertainty-Aware Multi-Task Loss  (equation 7)
# ---------------------------------------------------------------------------
