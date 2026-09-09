# Detection-path audit

Source inspection, 2026-09-09. No trained checkpoint is available locally, and
the local Python executable is inaccessible. Runtime gradient magnitudes and
checkpoint accuracy have not been verified.

## Confirmed implementation behavior

- `MultiviewMOTR._forward_single_image` encodes each image once. Independent
  detection and tracking decoders consume that memory in both train and eval.
- Detection runs fresh learned queries. Its final features replace the content
  half of the fresh tracking queries; carried queries keep their content. Clone
  and slice assignment do not intentionally detach detection features.
- **Reference handoff is not predicted-box handoff.** The detection decoder's
  `bbox_embed` remains `None`; only the tracking decoder gets iterative box
  heads. Consequently detection `det_refs` repeat its initial 2D anchors.
  Tracking is seeded from these anchors, not `det_pred_boxes` centers. Comments
  describing these references as refined are inaccurate. This is a confirmed
  implementation property, not proof of the observed low recall's cause.
- Detection logits and boxes use the last tracking classification/box heads.
  Heads are shared, although decoder parameters are separate. Detection has
  final-layer supervision; tracking also has auxiliary-layer supervision.
- `ClipMatcher.match_for_single_frame` independently matches detection outputs
  against all frame GT and computes classification/L1/GIoU losses. Their keys
  are registered in the build-time weight dictionary.
- The uncertainty objective includes detection and tracking terms. The copies
  returned for component logging are detached *after* constructing the objective;
  this does not disconnect the objective. Source inspection found no deliberate
  detach between detection features and their detection loss or tracking content.
- Fresh queries come first in QIM output, consistent with the handoff's indexing.
- Training assigns IDs using GT matching and selects carried tracks by ID/IoU;
  inference assigns IDs by confidence and selects tracks by ID. This is a real
  train/inference distinction, but not automatically a bug.
- The demo renders tracking decoder boxes, not detection decoder boxes. Queries
  without local IDs are filtered by QIM. Cross-camera matches are not required
  for local boxes to be displayed.
- Backbone construction requests ImageNet initialization on the main process.
  No MOTR checkpoint does not imply every network parameter started randomly.

## Next checkpoint experiment

Run `tools/audit_fusiontrack_demo.py` with the same arguments as the normal demo,
plus `--audit_dir /kaggle/working/inference_audit --audit_frames 3`. It observes
the existing inference call and saves both decoders' raw predictions, score
counts, and overlays before ID filtering. It does not change model behavior.
Only the first three synchronized sets are recorded; normal demo processing
continues for the full scene. Use a short scene for a quick run.

Overlays show up to 100 predictions above 0.3, labeled by query index, not object
ID. JSON contains every query. Camera indices follow `--camera_names` order.
Images are at model input resolution, so compare normalized boxes or resize GT
before numerical comparisons.

- Poor raw detection and tracking: examine actual loss components, annotation
  coverage, and gradients on a labeled training sample before changing training.
- Good detection but poor raw tracking: investigate anchor/content handoff and
  shared heads; compare identical inputs with the trained weights.
- Good raw tracking but sparse demo: inspect ID thresholds and QIM selection.

Existing regression tests cover shared encoder differentiation and checkpoint
gradient parity in smaller components. They do not establish full-model
detection-loss gradients or trained accuracy. A real labeled backward pass is
still required to measure detection-decoder, tracking-decoder, and shared-head
gradient norms independently. Do not treat this static audit as that test.
