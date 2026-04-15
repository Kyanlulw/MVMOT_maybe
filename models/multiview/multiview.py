# ------------------------------------------------------------------------
# Multiview extension of MOTR
# Each camera view is processed independently, outputting independent
# single-view track queries per camera.
# ------------------------------------------------------------------------
# Based on MOTR (https://github.com/megvii-model/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Multiview MOTR: multi-camera input with independent per-view tracking.

Design principles:
- Shared backbone, transformer, class_embed, bbox_embed weights across cameras.
- Each camera maintains its own track_instances (queries, obj_ids, memory, etc.).
- Forward pass loops over cameras independently — no cross-view information.
- Output: dict mapping cam_idx -> track_instances (single-view tracks).
- QueueMemoryBank is per-camera (already supported by the original QueueMemoryBank).
"""

import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Optional, Tuple

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from ..backbone import build_backbone
from ..matcher import build_matcher
from ..deformable_transformer_plus import build_deforamble_transformer
from ..qim import build as build_query_interaction_layer
from ..memory_bank import build_memory_bank
from ..temp import QueueMemoryBank
from ..deformable_detr import SetCriterion, MLP
from ..segmentation import sigmoid_focal_loss
from ..reid_query import ReIDQueryModule, build_reid_query_vit_model

# Re-use ClipMatcher unchanged — it is instantiated once per camera during training.
from ..motr import ClipMatcher, TrackerPostProcess, RuntimeTrackerBase, _get_clones


# ---------------------------------------------------------------------------
# Multiview ClipMatcher wrapper
# ---------------------------------------------------------------------------

class MultiviewClipMatcher(nn.Module):
    """
    Wraps one ClipMatcher per camera so that losses are computed independently
    for each view and then aggregated.
    """

    def __init__(
        self,
        num_cams: int,
        num_classes: int,
        matcher,
        weight_dict: dict,
        losses: list,
        use_uncertainty_loss: bool = False,
        uncertainty_init_tracking: float = -1.85,
        uncertainty_init_reid: float = -1.05,
    ):
        super().__init__()
        self.num_cams = num_cams
        # One criterion per camera (they share the same config but keep separate state).
        self.criteria: List[ClipMatcher] = nn.ModuleList(
            [
                ClipMatcher(
                    num_classes,
                    matcher,
                    weight_dict,
                    losses,
                    use_uncertainty_loss=use_uncertainty_loss,
                    uncertainty_init_tracking=uncertainty_init_tracking,
                    uncertainty_init_reid=uncertainty_init_reid,
                )
                for _ in range(num_cams)
            ]
        )

        per_cam_weight_dict = self.criteria[0].weight_dict if num_cams > 0 else {}
        self.weight_dict = {
            f"cam{cam_idx}_{key}": value
            for cam_idx in range(num_cams)
            for key, value in per_cam_weight_dict.items()
        }

    def initialize_for_single_clip(self, gt_instances_per_cam: List[List[Instances]]):
        """
        Args:
            gt_instances_per_cam: list[cam] of list[frame] of Instances
        """
        assert len(gt_instances_per_cam) == self.num_cams
        for cam_idx, criterion in enumerate(self.criteria):
            criterion.initialize_for_single_clip(gt_instances_per_cam[cam_idx])

    def get_criterion(self, cam_idx: int) -> ClipMatcher:
        return self.criteria[cam_idx]

    def forward(self, outputs_per_cam: Dict[int, dict]) -> dict:
        """
        Aggregate losses from all cameras.

        Args:
            outputs_per_cam: dict cam_idx -> {'losses_dict': ..., 'num_samples': int}

        Returns:
            Flat loss dict (losses averaged over cameras).
        """
        aggregated = {}
        for cam_idx, cam_out in outputs_per_cam.items():
            criterion = self.criteria[cam_idx]
            losses = cam_out.pop("losses_dict")
            num_samples = criterion.get_num_boxes(criterion.num_samples)

            # Match ClipMatcher.forward behaviour: normalize per-camera losses first,
            # then optionally compose uncertainty-weighted objective.
            losses = {loss_name: (loss_val / num_samples) for loss_name, loss_val in losses.items()}
            if criterion.use_uncertainty_loss:
                losses = criterion._append_uncertainty_loss(losses)

            for loss_name, loss_val in losses.items():
                key = f"cam{cam_idx}_{loss_name}"
                aggregated[key] = loss_val
        return aggregated


# ---------------------------------------------------------------------------
# MultiviewMOTR
# ---------------------------------------------------------------------------

class MultiviewMOTR(nn.Module):
    """
    Multi-camera MOTR where each camera is tracked independently.

    Each camera view shares the same backbone / transformer / head weights,
    but maintains its own:
      - track_instances  (queries, embeddings, obj_ids, memory, ...)
      - RuntimeTrackerBase (for inference ID assignment)
      - QueueMemoryBank slot (indexed by cam_idx)

    Input (training):
        data = {
            'imgs':        list[cam] of list[frame Tensor],   # C x T tensors
            'gt_instances': list[cam] of list[frame Instances]
        }

    Input (inference):
        Provided per-camera via `inference_single_image`.

    Output (training):
        {
            'losses_dict': flat dict of cam{i}_frame_{j}_loss_* keys
        }

    Output (inference, per call):
        {
            'track_instances': Instances,   # for the queried camera
            'ref_pts': Tensor (optional)
        }
    """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes: int,
        num_queries: int,
        num_feature_levels: int,
        criterion: MultiviewClipMatcher,
        track_embed,
        num_cams: int = 1,
        aux_loss: bool = True,
        with_box_refine: bool = False,
        two_stage: bool = False,
        memory_bank=None,
        use_checkpoint: bool = False,
        track_query_history_len: int = 0,
        track_query_queue_score_thresh: float = 0.0,
        use_reid_query: bool = False,
        reid_num_ids: int = 2048,
        reid_tau1: int = 10,
        reid_tau2: int = 4,
        reid_label_smoothing: float = 0.1,
        reid_vit_dim: int = 256,
        reid_num_layers: int = 2,
        reid_num_heads: int = 8,
        reid_dropout: float = 0.1,
        reid_temporal_decay_alpha: float = 1.0,
        cross_view_reid_match_thresh: float = 0.7,
        cross_view_reid_momentum: float = 0.9,
    ):
        super().__init__()
        self.num_cams = num_cams
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        self.reid = None
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint

        # Input projection layers (shared across cameras)
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Head initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # Per-camera post-processing and track management
        self.post_process = TrackerPostProcess()
        # One RuntimeTrackerBase per camera for inference
        self.track_bases: List[RuntimeTrackerBase] = [
            RuntimeTrackerBase() for _ in range(num_cams)
        ] if num_cams > 0 else []

        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

        # Per-camera queue memory banks
        self.track_query_queue_len = max(0, int(track_query_history_len))
        self.track_query_queue: Optional[QueueMemoryBank] = None
        self._queue_frame_idx: List[int] = [0] * num_cams
        if self.track_query_queue_len > 0:
            # QueueMemoryBank supports multiple cameras natively via cam_idx
            self.track_query_queue = QueueMemoryBank(
                num_cams=num_cams,
                maxlen=self.track_query_queue_len,
                score_thresh=float(track_query_queue_score_thresh),
            )

        self.use_reid_query = bool(use_reid_query)
        self.reid_num_ids = max(1, int(reid_num_ids))
        self.tmp_window_tau1 = max(1, int(reid_tau1))
        self._reid_obj_to_cls: List[Dict[int, int]] = [dict() for _ in range(num_cams)]
        self.reid_queue_bank: Optional[QueueMemoryBank] = None
        self.reid_module: Optional[ReIDQueryModule] = None
        if self.use_reid_query:
            track_dim = hidden_dim
            reid_dim = int(reid_vit_dim) if int(reid_vit_dim) > 0 else track_dim
            reid_transformer_type = (
                'deit_small_patch16_224_TransReID'
                if reid_dim <= 384 else
                'vit_base_patch16_224_TransReID'
            )

            reid_model = build_reid_query_vit_model(
                transformer_type=reid_transformer_type,
                drop_rate=float(reid_dropout),
                attn_drop_rate=float(reid_dropout),
                drop_path_rate=0.1,
                img_size=(256, 128),
                stride_size=16,
                camera_num=0,
                view_num=0,
                sie_xishu=1.0,
            )

            self.reid_module = ReIDQueryModule(
                reid_model=reid_model,
                track_dim=track_dim,
                num_ids=self.reid_num_ids,
                tau1=max(1, int(reid_tau1)),
                tau2=max(1, int(reid_tau2)),
                temporal_decay_alpha=float(reid_temporal_decay_alpha),
                label_smoothing=float(reid_label_smoothing),
            )
            self.reid_queue_bank = QueueMemoryBank(
                num_cams=num_cams,
                maxlen=max(1, int(reid_tau1)),
                score_thresh=float(track_query_queue_score_thresh),
            )

        # Cross-view ReID state (inference-time global identity association).
        self.cross_view_reid_match_thresh = float(cross_view_reid_match_thresh)
        self.cross_view_reid_momentum = float(cross_view_reid_momentum)
        self._cross_view_local_to_global: Dict[Tuple[int, int], int] = {}
        self._cross_view_global_tracks: Dict[int, List[Tuple[int, int]]] = {}
        self._cross_view_global_proto: Dict[int, Tensor] = {}
        self._next_cross_view_global_id: int = 0
        # TMP entry state per (camera, global_id): last seen frame, active/inactive, current local id.
        self._tmp_track_state: Dict[Tuple[int, int], Dict[str, int]] = {}

        # Paper-aligned behavior: keep unmatched queries inactive within tau1 before drop.
        for tb in self.track_bases:
            tb.miss_tolerance = self.tmp_window_tau1

    # ------------------------------------------------------------------
    # Per-camera track instance management
    # ------------------------------------------------------------------

    def _generate_empty_tracks(self) -> Instances:
        """Generate fresh, empty track queries (shared structure, called per camera)."""
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape
        device = self.query_embed.weight.device

        track_instances.ref_pts = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances),), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances),), dtype=torch.float32, device=device)

        return track_instances.to(device)

    def _reset_track_query_queue(self, cam_idx: Optional[int] = None):
        """Reset queue for one camera or all cameras."""
        if cam_idx is None:
            self._queue_frame_idx = [0] * self.num_cams
            if self.track_query_queue is not None:
                self.track_query_queue.clear()
            if self.reid_queue_bank is not None:
                self.reid_queue_bank.clear()
            if self.use_reid_query:
                self._reid_obj_to_cls = [dict() for _ in range(self.num_cams)]
            self._cross_view_local_to_global.clear()
            self._cross_view_global_tracks.clear()
            self._cross_view_global_proto.clear()
            self._next_cross_view_global_id = 0
            self._tmp_track_state.clear()
        else:
            self._queue_frame_idx[cam_idx] = 0
            if self.track_query_queue is not None:
                self.track_query_queue.clear(cam_idx=cam_idx)
            if self.reid_queue_bank is not None:
                self.reid_queue_bank.clear(cam_idx=cam_idx)
            if self.use_reid_query:
                self._reid_obj_to_cls[cam_idx].clear()

            remove_keys = [key for key in self._cross_view_local_to_global if key[0] == cam_idx]
            for key in remove_keys:
                gid = self._cross_view_local_to_global.pop(key)
                if gid in self._cross_view_global_tracks:
                    self._cross_view_global_tracks[gid] = [
                        pair for pair in self._cross_view_global_tracks[gid] if pair != key
                    ]
                    if len(self._cross_view_global_tracks[gid]) == 0:
                        self._cross_view_global_tracks.pop(gid, None)
                        self._cross_view_global_proto.pop(gid, None)

            stale_tmp = [key for key in self._tmp_track_state if key[0] == cam_idx]
            for key in stale_tmp:
                self._tmp_track_state.pop(key, None)

    def _build_reid_target_ids(self, cam_idx: int, track_instances: Instances) -> Optional[Tensor]:
        alive_mask = track_instances.obj_idxes >= 0
        alive_obj_ids = track_instances.obj_idxes[alive_mask]
        if len(alive_obj_ids) == 0:
            return None

        id_map = self._reid_obj_to_cls[cam_idx]
        targets = []
        for obj_id in alive_obj_ids.tolist():
            oid = int(obj_id)
            if oid not in id_map:
                id_map[oid] = len(id_map) % self.reid_num_ids
            targets.append(id_map[oid])

        return torch.as_tensor(targets, dtype=torch.long, device=track_instances.obj_idxes.device)

    def _update_track_query_queue(self, track_instances: Instances, cam_idx: int):
        if self.track_query_queue is None:
            return
        self.track_query_queue.push(
            cam_idx=cam_idx,
            track_instances=track_instances,
            frame_id=self._queue_frame_idx[cam_idx],
        )
        self._queue_frame_idx[cam_idx] += 1

    def get_track_query_queues(self, cam_idx: int = 0):
        if self.track_query_queue is None:
            return {}
        return self.track_query_queue.get_all_queues(cam_idx)

    def clear(self, cam_idx: Optional[int] = None):
        """Clear tracker state for one or all cameras."""
        if cam_idx is None:
            for tb in self.track_bases:
                tb.clear()
            self._reset_track_query_queue()
        else:
            self.track_bases[cam_idx].clear()
            self._reset_track_query_queue(cam_idx)

    def _prune_inference_tmp(self, current_frame_by_cam: Dict[int, int]):
        """Prune stale TMP/reid queue entries older than tau1 frames."""
        tau1 = self.tmp_window_tau1

        for cam_idx, current_frame in current_frame_by_cam.items():
            if self.reid_queue_bank is not None:
                for track_id, track_queue in list(self.reid_queue_bank.get_all_queues(cam_idx).items()):
                    if len(track_queue) == 0:
                        self.reid_queue_bank.remove_track(cam_idx, track_id)
                        continue
                    last_frame = int(track_queue.snapshots[-1].frame_id)
                    if (current_frame - last_frame) > tau1:
                        self.reid_queue_bank.remove_track(cam_idx, track_id)

            for key, state in list(self._tmp_track_state.items()):
                key_cam, gid = key
                if key_cam != cam_idx:
                    continue
                if (current_frame - int(state['last_seen'])) <= tau1:
                    continue

                # Discard TMP entry when undetected for > tau1.
                self._tmp_track_state.pop(key, None)

                remove_local_keys = [
                    local_key
                    for local_key, mapped_gid in self._cross_view_local_to_global.items()
                    if local_key[0] == cam_idx and mapped_gid == gid
                ]
                for local_key in remove_local_keys:
                    self._cross_view_local_to_global.pop(local_key, None)

                if gid in self._cross_view_global_tracks:
                    self._cross_view_global_tracks[gid] = [
                        pair for pair in self._cross_view_global_tracks[gid] if pair[0] != cam_idx
                    ]
                    if len(self._cross_view_global_tracks[gid]) == 0:
                        self._cross_view_global_tracks.pop(gid, None)
                        self._cross_view_global_proto.pop(gid, None)

    def _update_tmp_track_state(
        self,
        track_instances_by_cam: Dict[int, Instances],
        cross_view_ids: Dict[int, Tensor],
        current_frame_by_cam: Dict[int, int],
    ):
        """Update active/inactive TMP states and handle reactivation/new entries."""
        for cam_idx, track_instances in track_instances_by_cam.items():
            current_frame = current_frame_by_cam[cam_idx]
            observed_keys = set()

            if len(track_instances) > 0:
                valid = track_instances.obj_idxes >= 0
                if track_instances.has('scores'):
                    valid = valid & (track_instances.scores >= self.track_bases[cam_idx].filter_score_thresh)
                valid_indices = valid.nonzero(as_tuple=False).squeeze(1)

                for idx in valid_indices.tolist():
                    gid = int(cross_view_ids[cam_idx][idx].item())
                    if gid < 0:
                        continue

                    local_id = int(track_instances.obj_idxes[idx].item())
                    key = (cam_idx, gid)
                    prev_state = self._tmp_track_state.get(key)
                    if prev_state is None:
                        # New object: create a TMP entry.
                        self._tmp_track_state[key] = {
                            'last_seen': current_frame,
                            'active': 1,
                            'local_id': local_id,
                        }
                    else:
                        # Reactivation (if was inactive) or normal active update.
                        prev_state['last_seen'] = current_frame
                        prev_state['active'] = 1
                        prev_state['local_id'] = local_id

                    observed_keys.add(key)

            # Undetected entries remain in TMP as inactive if still inside tau1.
            for key, state in self._tmp_track_state.items():
                if key[0] != cam_idx:
                    continue
                if key in observed_keys:
                    continue
                state['active'] = 0

    def _export_tmp_status(self) -> Dict[str, Dict[str, int]]:
        status = {}
        for (cam_idx, gid), state in self._tmp_track_state.items():
            status[f"cam{cam_idx}_gid{gid}"] = {
                'active': int(state.get('active', 0)),
                'last_seen': int(state.get('last_seen', -1)),
                'local_id': int(state.get('local_id', -1)),
            }
        return status

    def _associate_cross_view_reid(self, track_instances_by_cam: Dict[int, Instances]) -> Dict[int, Tensor]:
        """Assign global IDs across cameras from per-track ReID embeddings."""
        cross_view_ids: Dict[int, Tensor] = {}
        used_gids_per_cam: Dict[int, set] = {}

        for cam_idx, track_instances in track_instances_by_cam.items():
            device = track_instances.obj_idxes.device
            cross_view_ids[cam_idx] = torch.full(
                (len(track_instances),), -1, dtype=torch.long, device=device
            )
            used_gids_per_cam[cam_idx] = set()

            if len(track_instances) == 0 or not track_instances.has('output_embedding'):
                continue

            valid = track_instances.obj_idxes >= 0
            if track_instances.has('scores'):
                valid = valid & (track_instances.scores >= self.track_bases[cam_idx].filter_score_thresh)
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)

            for track_idx in valid_indices.tolist():
                local_id = int(track_instances.obj_idxes[track_idx].item())
                local_key = (cam_idx, local_id)

                emb = track_instances.output_embedding[track_idx].detach()
                if emb.ndim != 1:
                    emb = emb.flatten()
                emb = F.normalize(emb, dim=0)

                if local_key in self._cross_view_local_to_global:
                    gid = self._cross_view_local_to_global[local_key]
                else:
                    gid = -1
                    best_sim = -1.0
                    for cand_gid, proto in self._cross_view_global_proto.items():
                        if cand_gid in used_gids_per_cam[cam_idx]:
                            continue
                        sim = torch.dot(emb, proto).item()
                        if sim > best_sim:
                            best_sim = sim
                            gid = cand_gid

                    if gid >= 0 and best_sim >= self.cross_view_reid_match_thresh:
                        self._cross_view_local_to_global[local_key] = gid
                        self._cross_view_global_tracks.setdefault(gid, []).append(local_key)
                    else:
                        gid = self._next_cross_view_global_id
                        self._next_cross_view_global_id += 1
                        self._cross_view_local_to_global[local_key] = gid
                        self._cross_view_global_tracks[gid] = [local_key]
                        self._cross_view_global_proto[gid] = emb

                if gid not in self._cross_view_global_proto:
                    self._cross_view_global_proto[gid] = emb
                else:
                    m = self.cross_view_reid_momentum
                    proto = self._cross_view_global_proto[gid]
                    self._cross_view_global_proto[gid] = F.normalize(m * proto + (1.0 - m) * emb, dim=0)

                used_gids_per_cam[cam_idx].add(gid)
                cross_view_ids[cam_idx][track_idx] = gid

        return cross_view_ids

    def _export_cross_view_matches(self) -> Dict[int, List[Tuple[int, int]]]:
        matches: Dict[int, List[Tuple[int, int]]] = {}
        for gid, pairs in self._cross_view_global_tracks.items():
            unique_pairs = sorted(set((int(c), int(l)) for c, l in pairs))
            matches[int(gid)] = unique_pairs
        return matches

    # ------------------------------------------------------------------
    # Core single-image forward (reused for every camera/frame)
    # ------------------------------------------------------------------

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples: NestedTensor, track_instances: Instances) -> dict:
        """Run backbone + transformer for a single image + track queries."""
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)

        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'ref_pts': ref_pts_all[5],
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(
        self,
        frame_res: dict,
        track_instances: Instances,
        is_last: bool,
        cam_idx: int,
        global_frame_idx: Optional[int] = None,
    ) -> dict:
        """
        Post-process one frame for one camera.
        cam_idx is needed to route to the correct criterion / track_base.
        """
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]

        if self.training:
            frame_res['track_instances'] = track_instances
            # Use the per-camera criterion for matching
            track_instances = self.criterion.get_criterion(cam_idx).match_for_single_frame(frame_res)
        else:
            self.track_bases[cam_idx].update(track_instances)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            if self.training:
                self.criterion.get_criterion(cam_idx).calc_loss_for_track_scores(track_instances)

        if self.reid_module is not None and self.reid_queue_bank is not None:
            target_ids = self._build_reid_target_ids(cam_idx, track_instances) if self.training else None
            reid_frame_idx = self._queue_frame_idx[cam_idx] if global_frame_idx is None else int(global_frame_idx)
            track_instances, reid_loss = self.reid_module(
                track_instances=track_instances,
                queue_bank=self.reid_queue_bank,
                cam_idx=cam_idx,
                global_frame_idx=reid_frame_idx,
                target_ids=target_ids,
            )

            if self.training and reid_loss is not None:
                criterion_cam = self.criterion.get_criterion(cam_idx)
                frame_idx = max(0, criterion_cam._current_frame_idx - 1)
                if isinstance(reid_loss, dict):
                    if 'total' in reid_loss:
                        criterion_cam.losses_dict[f'frame_{frame_idx}_reid_total'] = reid_loss['total']
                    if 'ce' in reid_loss:
                        criterion_cam.losses_dict[f'frame_{frame_idx}_reid_ce'] = reid_loss['ce']
                    if 'triplet' in reid_loss:
                        criterion_cam.losses_dict[f'frame_{frame_idx}_reid_triplet'] = reid_loss['triplet']
                else:
                    criterion_cam.losses_dict[f'frame_{frame_idx}_reid_total'] = reid_loss

        self._update_track_query_queue(track_instances, cam_idx)

        tmp = {
            'init_track_instances': self._generate_empty_tracks(),
            'track_instances': track_instances,
        }
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    # ------------------------------------------------------------------
    # Per-camera clip forward (training)
    # ------------------------------------------------------------------

    def _forward_single_camera_clip(
        self,
        frames: List[Tensor],
        cam_idx: int,
        global_frame_idxs: Optional[List[int]] = None,
    ) -> Tuple[dict, Instances]:
        """
        Process all frames of one camera sequentially.

        Returns:
            outputs: dict with pred_logits/pred_boxes lists
            track_instances: final track state for this camera
        """
        outputs = {'pred_logits': [], 'pred_boxes': []}
        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())

        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            is_last = (frame_index == len(frames) - 1)
            frame_global_idx = None
            if global_frame_idxs is not None and frame_index < len(global_frame_idxs):
                frame_global_idx = int(global_frame_idxs[frame_index])

            if self.use_checkpoint and frame_index < len(frames) - 2:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                    )

                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple(p for p in self.parameters() if p.requires_grad)
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'hs': tmp[3],
                    'aux_outputs': [{'pred_logits': tmp[4 + i], 'pred_boxes': tmp[4 + 5 + i]} for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances)

            frame_res = self._post_process_single_image(
                frame_res,
                track_instances,
                is_last,
                cam_idx,
                global_frame_idx=frame_global_idx,
            )
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        return outputs, track_instances

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_single_image(
        self,
        img,
        ori_img_size: Tuple[int, int],
        cam_idx: int = 0,
        track_instances: Optional[Instances] = None,
    ) -> dict:
        """
        Run inference for a single image on the given camera.

        Args:
            img: Tensor or NestedTensor for the frame.
            ori_img_size: (H, W) of the original image.
            cam_idx: which camera this image belongs to.
            track_instances: carry-over track state from the previous frame.
                             If None, resets the camera's queue and starts fresh.

        Returns:
            {
                'track_instances': Instances,
                'ref_pts': Tensor (optional)
            }
        """
        assert 0 <= cam_idx < self.num_cams, f"cam_idx {cam_idx} out of range [0, {self.num_cams})"
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            self._reset_track_query_queue(cam_idx)
            track_instances = self._generate_empty_tracks()

        res = self._forward_single_image(img, track_instances)
        res = self._post_process_single_image(res, track_instances, False, cam_idx)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)

        current_frame_by_cam = {cam_idx: max(0, self._queue_frame_idx[cam_idx] - 1)}
        self._prune_inference_tmp(current_frame_by_cam)

        cross_view_ids = self._associate_cross_view_reid({cam_idx: track_instances})
        self._update_tmp_track_state(
            {cam_idx: track_instances},
            cross_view_ids,
            current_frame_by_cam,
        )
        track_instances.cross_view_ids = cross_view_ids[cam_idx]

        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ret['ref_pts'] = ref_pts * scale_fct[None]
        ret['cross_view_matches'] = self._export_cross_view_matches()
        ret['tmp_status'] = self._export_tmp_status()
        return ret

    @torch.no_grad()
    def inference_single_image_multiview(
        self,
        imgs: List[Tensor],
        ori_img_sizes: List[Tuple[int, int]],
        track_instances_list: Optional[List[Optional[Instances]]] = None,
    ) -> dict:
        """Run synchronized inference for all cameras and return cross-view ReID matches."""
        assert len(imgs) == self.num_cams, f"Expected {self.num_cams} images, got {len(imgs)}"
        assert len(ori_img_sizes) == self.num_cams, f"Expected {self.num_cams} image sizes, got {len(ori_img_sizes)}"

        if track_instances_list is None:
            track_instances_list = [None] * self.num_cams
        else:
            assert len(track_instances_list) == self.num_cams, \
                f"Expected {self.num_cams} track states, got {len(track_instances_list)}"

        views = []
        track_instances_by_cam: Dict[int, Instances] = {}
        current_frame_by_cam: Dict[int, int] = {}

        for cam_idx in range(self.num_cams):
            img = imgs[cam_idx]
            if not isinstance(img, NestedTensor):
                img = nested_tensor_from_tensor_list(img)

            track_instances = track_instances_list[cam_idx]
            if track_instances is None:
                self._reset_track_query_queue(cam_idx)
                track_instances = self._generate_empty_tracks()

            res = self._forward_single_image(img, track_instances)
            res = self._post_process_single_image(res, track_instances, False, cam_idx)

            post_track_instances = self.post_process(res['track_instances'], ori_img_sizes[cam_idx])
            track_instances_by_cam[cam_idx] = post_track_instances
            current_frame_by_cam[cam_idx] = max(0, self._queue_frame_idx[cam_idx] - 1)

            view_out = {'track_instances': post_track_instances}
            if 'ref_pts' in res:
                ref_pts = res['ref_pts']
                img_h, img_w = ori_img_sizes[cam_idx]
                scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
                view_out['ref_pts'] = ref_pts * scale_fct[None]
            views.append(view_out)

        self._prune_inference_tmp(current_frame_by_cam)
        cross_view_ids = self._associate_cross_view_reid(track_instances_by_cam)
        self._update_tmp_track_state(track_instances_by_cam, cross_view_ids, current_frame_by_cam)
        for cam_idx in range(self.num_cams):
            views[cam_idx]['track_instances'].cross_view_ids = cross_view_ids[cam_idx]

        return {
            'views': views,
            'cross_view_matches': self._export_cross_view_matches(),
            'tmp_status': self._export_tmp_status(),
        }

    def forward(self, data: dict) -> dict:
        """
        Training forward over a full clip for all cameras.

        Args:
            data: {
                'imgs':         list[cam_idx] of list[frame Tensor],
                'gt_instances': list[cam_idx] of list[frame Instances],
            }

        Returns:
            {'losses_dict': flat dict of cam{i}_frame_{j}_loss_* values}
        """
        assert self.training, "Use inference_single_image for inference."

        imgs_per_cam: List[List[Tensor]] = data.get('imgs_multiview', data['imgs'])
        gt_per_cam: List[List[Instances]] = data.get('gt_instances_multiview', data['gt_instances'])
        global_frame_idxs: Optional[List[int]] = data.get('global_frame_idxs', None)
        assert len(imgs_per_cam) == self.num_cams, \
            f"Expected {self.num_cams} cameras, got {len(imgs_per_cam)}"

        self.criterion.initialize_for_single_clip(gt_per_cam)

        outputs_per_cam: Dict[int, dict] = {}
        for cam_idx in range(self.num_cams):
            self._reset_track_query_queue(cam_idx)
            cam_outputs, _ = self._forward_single_camera_clip(
                imgs_per_cam[cam_idx],
                cam_idx,
                global_frame_idxs=global_frame_idxs,
            )
            cam_outputs['losses_dict'] = self.criterion.get_criterion(cam_idx).losses_dict
            outputs_per_cam[cam_idx] = cam_outputs

        losses = self.criterion(outputs_per_cam)
        return {'losses_dict': losses}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
        'e2e_mv_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    # Number of cameras — falls back to 1 for single-view compatibility.
    num_cams = getattr(args, 'num_cams', 1)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(
        args, args.query_interaction_layer, d_model, hidden_dim, d_model * 2
    )

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)

    # Build per-frame weight dict (same structure as original MOTR)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({
            f"frame_{i}_loss_ce":   args.cls_loss_coef,
            f"frame_{i}_loss_bbox": args.bbox_loss_coef,
            f"frame_{i}_loss_giou": args.giou_loss_coef,
        })
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({
                    f"frame_{i}_aux{j}_loss_ce":   args.cls_loss_coef,
                    f"frame_{i}_aux{j}_loss_bbox": args.bbox_loss_coef,
                    f"frame_{i}_aux{j}_loss_giou": args.giou_loss_coef,
                })

    if getattr(args, 'use_reid_query', False):
        for i in range(num_frames_per_batch):
            weight_dict.update({f"frame_{i}_reid_total": args.reid_loss_coef})

    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({f"frame_{i}_track_loss_ce": args.cls_loss_coef})
    else:
        memory_bank = None

    losses = ['labels', 'boxes']
    shared_uncertainty_init = getattr(args, 'uncertainty_init', None)
    uncertainty_init_tracking = getattr(args, 'uncertainty_init_tracking', -1.85)
    uncertainty_init_reid = getattr(args, 'uncertainty_init_reid', -1.05)
    if shared_uncertainty_init is not None:
        uncertainty_init_tracking = float(shared_uncertainty_init)
        uncertainty_init_reid = float(shared_uncertainty_init)

    criterion = MultiviewClipMatcher(
        num_cams=num_cams,
        num_classes=num_classes,
        matcher=img_matcher,
        weight_dict=weight_dict,
        losses=losses,
        use_uncertainty_loss=getattr(args, 'use_uncertainty_loss', False),
        uncertainty_init_tracking=uncertainty_init_tracking,
        uncertainty_init_reid=uncertainty_init_reid,
    )
    criterion.to(device)

    track_query_queue_len = getattr(args, 'track_query_queue_len',
                                    getattr(args, 'track_query_history_len', 0))
    track_query_queue_score_thresh = getattr(args, 'track_query_queue_score_thresh', 0.0)

    model = MultiviewMOTR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        criterion=criterion,
        track_embed=query_interaction_layer,
        num_cams=num_cams,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        track_query_history_len=track_query_queue_len,
        track_query_queue_score_thresh=track_query_queue_score_thresh,
        use_reid_query=getattr(args, 'use_reid_query', False),
        reid_num_ids=getattr(args, 'reid_num_ids', 2048),
        reid_tau1=getattr(args, 'reid_tau1', 10),
        reid_tau2=getattr(args, 'reid_tau2', 4),
        reid_label_smoothing=getattr(args, 'reid_label_smoothing', 0.1),
        reid_vit_dim=getattr(args, 'reid_vit_dim', hidden_dim),
        reid_num_layers=getattr(args, 'reid_num_layers', 2),
        reid_num_heads=getattr(args, 'reid_num_heads', 8),
        reid_dropout=getattr(args, 'reid_dropout', 0.1),
        reid_temporal_decay_alpha=getattr(args, 'reid_temporal_decay_alpha', 1.0),
        cross_view_reid_match_thresh=getattr(args, 'cross_view_reid_match_thresh', 0.7),
        cross_view_reid_momentum=getattr(args, 'cross_view_reid_momentum', 0.9),
    )
    model.to(device)

    postprocessors = {}
    return model, criterion, postprocessors