# FusionTrack inference without OUM

Use `demo_multiview.py` for synchronized cross-view inference. It restores saved
training arguments from `--resume`, allows explicit CLI overrides, and loads
weights strictly. Without checkpoint metadata its architecture default is
`fusiontrack_motr`. An incompatible configuration fails instead of silently
dropping the detection decoder or initializing missing weights.

Example (replace paths):

```sh
python demo_multiview.py --resume /path/to/checkpoint.pth --scene_dir /path/to/scene --camera_names cam1 cam2 cam3 cam4 cam5 --device cuda --output_video tracks.avi
```

Each camera folder must contain `images/`. Frame filename stems must identify
the same timestamps across cameras. Numeric stems sort numerically; mismatched
sets fail instead of pairing frames by directory position. TMP age is measured
in processed synchronized frames, not numeric filename differences. Call
`inference_single_image_multiview` once per synchronized frame and carry all
returned track states to the next call. Passing `None` starts a new sequence and
resets both local trackers and global memory.

## Association

Based on FusionTrack section IV-F, pages 8–9:

1. Preserve the learned trajectory ReID descriptors through the query handoff.
2. Mask same-camera pairs and select mutual top-k candidates across **all** other
   cameras, with the cosine similarity confidence threshold applied.
3. Filter pairs by spatial-neighbor correspondence, counting distinct matches.
4. Cluster with at most one object per view in each identity.
5. Reconcile clusters with persistent IDs; reserve continuing IDs before matching
   newcomers. Retain inactive TMP entries through tau1 frames and prune them when
   their age exceeds tau1. Existing local tracks retain their shared global ID
   across temporary cross-view appearance disagreement when views do not collide.

## Explicit implementation choices

The paper does not specify every numerical setting or the hierarchical linkage.
This implementation keeps deterministic complete linkage, cosine similarity
threshold 0.8, top-k 10, five spatial neighbors, and a strict neighbor-support
threshold greater than 0.5. The last three settings are configurable with
`--cross_view_top_k`, `--cross_view_spatial_neighbors`, and
`--cross_view_neighbor_thresh`. Neighbor support uses maximum one-to-one matching
divided by the larger neighborhood size. If either neighborhood is empty,
appearance matching is retained because spatial evidence is unavailable.

Persistent-ID reconciliation and latest-descriptor re-entry lookup remain
engineering choices: the paper does not provide an exact reconciliation
algorithm. OUM cross-frame/cross-view query refinement remains intentionally
absent. This is therefore an OUM-free adaptation, not a claim of exact full-paper
reproduction. Raw tracking features are used only when ReID is disabled.

Regression coverage checks descriptor propagation, global top-k masking,
neighbor support, view exclusivity, identity continuity, TMP reactivation and
expiry, and input frame alignment. These checks do not establish tracking
accuracy; compare identity metrics on held-out sequences using a trained
checkpoint to measure the effect.
