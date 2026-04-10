# ------------------------------------------------------------------------
# queue_memory_bank.py
#
# A pure FIFO queue that stores (embedding, query_pos) snapshots per
# track per camera.  No attention, no learned layers — just storage.
# A separate ReID module consumes the queue for re-identification.
# ------------------------------------------------------------------------

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.structures import Instances


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackSnapshot:
    """One frame's worth of data for a single track."""
    embedding : Tensor          # (dim,)
    query_pos : Tensor          # (dim,)  or (pos_dim,)
    score     : float
    frame_id  : int


@dataclass
class TrackQueue:
    """
    A fixed-length FIFO queue of snapshots for one track in one camera.
    Oldest entry is automatically dropped when maxlen is reached.
    """
    track_id : int
    cam_idx  : int
    maxlen   : int
    snapshots: deque = field(default_factory=deque)

    def __post_init__(self):
        self.snapshots = deque(maxlen=self.maxlen)

    def push(self, snapshot: TrackSnapshot):
        self.snapshots.append(snapshot)

    def get_embeddings(self) -> Tensor:
        """Returns (L, dim) — all stored embeddings, oldest first."""
        return torch.stack([s.embedding for s in self.snapshots], dim=0)

    def get_query_pos(self) -> Tensor:
        """Returns (L, pos_dim)."""
        return torch.stack([s.query_pos for s in self.snapshots], dim=0)

    def __len__(self):
        return len(self.snapshots)


# ---------------------------------------------------------------------------
# Queue-based memory bank  (no nn.Module — pure data structure)
# ---------------------------------------------------------------------------

class QueueMemoryBank:
    """
    Maintains one TrackQueue per (camera, track_id) pair.

    Layout:
        _queues[cam_idx][track_id] -> TrackQueue

    This is intentionally NOT an nn.Module — it holds no parameters.
    Attach it to your tracker and call push() each frame.
    """

    def __init__(self, num_cams: int, maxlen: int, score_thresh: float = 0.5):
        self.num_cams     = num_cams
        self.maxlen       = maxlen
        self.score_thresh = score_thresh

        # dict-of-dicts:  cam_idx -> { track_id -> TrackQueue }
        self._queues: List[Dict[int, TrackQueue]] = [
            {} for _ in range(num_cams)
        ]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(
        self,
        cam_idx        : int,
        track_instances: Instances,
        frame_id       : int,
        score_thresh   : Optional[float] = None,
    ):
        """
        Push current-frame embeddings into the queue for one camera.

        Only tracks whose score exceeds score_thresh are stored,
        matching the original MemoryBank's save logic.

        Args:
            cam_idx:         which camera these instances came from
            track_instances: Instances with fields:
                               .output_embedding  (N, dim)
                               .query_pos         (N, pos_dim)
                               .scores            (N,)
                               .obj_idxes         (N,)  — stable track IDs
            frame_id:        current frame number (for bookkeeping)
            score_thresh:    override the default threshold
        """
        thresh = score_thresh if score_thresh is not None else self.score_thresh

        embeddings = track_instances.output_embedding   # (N, dim)
        query_pos  = track_instances.query_pos          # (N, pos_dim)
        scores     = track_instances.scores             # (N,)
        track_ids  = track_instances.obj_idxes          # (N,)  long tensor

        save_mask = scores > thresh

        for i, (tid, save) in enumerate(
                zip(track_ids.tolist(), save_mask.tolist())):
            if not save:
                continue

            tid = int(tid)
            if tid not in self._queues[cam_idx]:
                self._queues[cam_idx][tid] = TrackQueue(
                    track_id=tid,
                    cam_idx=cam_idx,
                    maxlen=self.maxlen,
                )

            snapshot = TrackSnapshot(
                embedding=embeddings[i].detach().cpu(),
                query_pos=query_pos[i].detach().cpu(),
                score=float(scores[i]),
                frame_id=frame_id,
            )
            self._queues[cam_idx][tid].push(snapshot)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_queue(self, cam_idx: int, track_id: int) -> Optional[TrackQueue]:
        """Fetch one track's queue.  Returns None if track not seen yet."""
        return self._queues[cam_idx].get(track_id, None)

    def get_all_queues(self, cam_idx: int) -> Dict[int, TrackQueue]:
        """All queues for one camera.  Used by ReID to build a gallery."""
        return self._queues[cam_idx]

    def get_all_cams(self) -> List[Dict[int, TrackQueue]]:
        """All queues across every camera."""
        return self._queues

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove_track(self, cam_idx: int, track_id: int):
        """Call this when a track is confirmed dead to free memory."""
        self._queues[cam_idx].pop(track_id, None)

    def remove_stale_tracks(self, cam_idx: int, active_ids: List[int]):
        """Drop any track not in active_ids (called each frame)."""
        stale = [
            tid for tid in self._queues[cam_idx]
            if tid not in active_ids
        ]
        for tid in stale:
            del self._queues[cam_idx][tid]

    def clear(self, cam_idx: Optional[int] = None):
        if cam_idx is not None:
            self._queues[cam_idx].clear()
        else:
            for c in range(self.num_cams):
                self._queues[c].clear()