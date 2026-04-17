# ------------------------------------------------------------------------
# Multi-View MOTR Demo
# ------------------------------------------------------------------------
# Inference script for multi-view multi-object tracking with overlapping
# cameras. Processes synchronized frames from multiple camera views and
# outputs per-view tracks with cross-view global ID association.
# ------------------------------------------------------------------------

import argparse
import math
import os
import os.path as osp
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from main import get_args_parser
from models import build_model
from models.structures import Instances
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.tool import load_model
import datasets.transforms as T


def make_inference_transforms():
    """Create transforms for inference."""
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.MotCompose([
        T.MotRandomResize([800], max_size=1333),
        normalize,
    ])


def discover_camera_names(scene_dir):
    """Discover camera folders that contain an images directory with frames."""
    cameras = []
    for name in sorted(os.listdir(scene_dir)):
        cam_dir = osp.join(scene_dir, name)
        img_dir = osp.join(cam_dir, 'images')
        if not osp.isdir(cam_dir) or not osp.isdir(img_dir):
            continue
        has_frames = any(
            f.lower().endswith(('.jpg', '.png', '.jpeg'))
            for f in os.listdir(img_dir)
        )
        if has_frames:
            cameras.append(name)
    return cameras


def collect_frame_files(scene_dir, camera_names):
    """Return sorted frame-file lists per camera and synchronized frame count."""
    frame_files_by_cam = {}
    min_frames = None
    for cam in camera_names:
        img_dir = osp.join(scene_dir, cam, 'images')
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        frame_files_by_cam[cam] = img_files
        if min_frames is None:
            min_frames = len(img_files)
        else:
            min_frames = min(min_frames, len(img_files))
    return frame_files_by_cam, (0 if min_frames is None else min_frames)


def load_multiview_frames(scene_dir, camera_names, frame_files_by_cam, frame_idx):
    """
    Load synchronized frames from multiple camera views.
    
    Args:
        scene_dir: Path to the scene directory
        camera_names: List of camera subdirectory names
        frame_idx: Frame index to load
        
    Returns:
        List of PIL images, List of original sizes
    """
    images = []
    ori_sizes = []
    for cam in camera_names:
        img_dir = osp.join(scene_dir, cam, 'images')
        img_files = frame_files_by_cam[cam]
        if frame_idx < len(img_files):
            img_path = osp.join(img_dir, img_files[frame_idx])
            img = Image.open(img_path)
            ori_sizes.append(img.size[::-1])  # (h, w)
            images.append(img)
    return images, ori_sizes


def draw_tracks_multiview(
    images, results, camera_names,
    global_id_mapping=None,
    color_map=None,
    score_thresh=0.5,
    box_thickness=1,
    font_scale=0.4,
):
    """
    Draw tracking results on multi-view images.
    
    Args:
        images: List of numpy images (H, W, 3) per view
        results: Dict with per-view track instances
        camera_names: List of camera names
        global_id_mapping: Optional global ID mapping for consistent colors
        color_map: Optional dict mapping track ID to color
    """
    if color_map is None:
        color_map = {}

    # Fallback table: (view_idx, local_id) -> global_id.
    # Primary source for alignment should be per-track cross_view_ids.
    local_to_global = {}
    if global_id_mapping:
        for gid, locals_list in global_id_mapping.items():
            gid_int = int(gid)
            for vid, lid in locals_list:
                local_to_global[(int(vid), int(lid))] = gid_int

    output_images = []
    for v, cam_name in enumerate(camera_names):
        img = images[v].copy()
        view_result = results['views'][v]
        track_instances = view_result['track_instances']

        if track_instances is not None and len(track_instances) > 0:
            boxes = track_instances.boxes.cpu().numpy() if hasattr(track_instances, 'boxes') else []
            scores = track_instances.scores.cpu().numpy() if hasattr(track_instances, 'scores') else []
            obj_ids = track_instances.obj_idxes.cpu().numpy() if hasattr(track_instances, 'obj_idxes') else []
            cross_view_ids = (
                track_instances.cross_view_ids.cpu().numpy()
                if hasattr(track_instances, 'cross_view_ids')
                else None
            )

            for i in range(len(boxes)):
                if scores[i] < score_thresh:
                    continue

                obj_id = int(obj_ids[i])
                if obj_id < 0:
                    continue

                # Prefer direct per-track cross-view assignment from model output,
                # fallback to mapping lookup if not present.
                global_id = -1
                if cross_view_ids is not None and i < len(cross_view_ids):
                    global_id = int(cross_view_ids[i])
                if global_id < 0:
                    global_id = int(local_to_global.get((v, obj_id), -1))

                # Use global ID for color if available, else local ID
                display_id = global_id if global_id >= 0 else obj_id
                if display_id not in color_map:
                    color_map[display_id] = tuple(
                        int(c) for c in np.random.randint(0, 255, 3)
                    )
                color = color_map[display_id]

                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(box_thickness)))

                # Label with both local and global ID
                if global_id >= 0:
                    label = f"G{global_id}|L{obj_id}"
                else:
                    label = f"L{obj_id}"
                label += f" {scores[i]:.2f}"

                text_thickness = max(1, int(box_thickness))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), text_thickness)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (255, 255, 255), text_thickness)

        # Add camera label
        cv2.putText(img, cam_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        output_images.append(img)

    return output_images, color_map


def create_video_writer(output_video, frame_size, fps=30):
    """Create a VideoWriter with codec/container fallbacks."""
    target = Path(output_video)
    ext = target.suffix.lower()

    candidates_by_ext = {
        '.mp4': [('MJPG', '.avi'), ('XVID', '.avi')],
        '.avi': [('MJPG', '.avi'), ('XVID', '.avi')],
        '.mkv': [('MJPG', '.avi'), ('XVID', '.avi')],
        'default': [('MJPG', '.avi'), ('XVID', '.avi')],
    }
    candidates = candidates_by_ext.get(ext, candidates_by_ext['default'])

    tried = []
    seen = set()
    for codec, out_ext in candidates:
        out_path = str(target.with_suffix(out_ext))
        key = (out_path, codec)
        if key in seen:
            continue
        seen.add(key)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer, out_path, codec

        writer.release()
        tried.append(f"{out_path} ({codec})")

    tried_str = ', '.join(tried)
    raise RuntimeError(
        f"Could not create video writer for {output_video}. Tried: {tried_str}"
    )


def main():
    parser = argparse.ArgumentParser('Multi-View MOTR Demo', parents=[get_args_parser()])
    parser.set_defaults(use_reid_query=True)
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Path to the scene directory with camera subdirectories')
    parser.add_argument('--camera_names', type=str, nargs='*', default=None,
                       help='Optional camera subdirectory names; if omitted, discover automatically')
    parser.add_argument('--output_video', type=str, default='multiview_output.avi',
                       help='Output video path (AVI recommended for compatibility)')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                       help='Score threshold for visualization')
    parser.add_argument('--output_fps', type=float, default=10.0,
                       help='Output video FPS (lower value gives slower playback)')
    parser.add_argument('--box_thickness', type=int, default=1,
                       help='Bounding box line thickness for visualization')
    parser.add_argument('--label_font_scale', type=float, default=0.4,
                       help='Label font scale for visualization')
    parser.add_argument('--no_use_reid_query', dest='use_reid_query', action='store_false',
                       help='Disable ReID query branch during demo inference')
    args = parser.parse_args()

    # Override some args for demo
    args.meta_arch = 'multiview_motr'
    args.dataset_file = 'e2e_mv_mot'
    if args.camera_names is None or len(args.camera_names) == 0:
        args.camera_names = discover_camera_names(args.scene_dir)
    if len(args.camera_names) == 0:
        raise ValueError(f"No camera folders with images found under: {args.scene_dir}")

    args.num_cams = len(args.camera_names)
    args.num_views = args.num_cams

    if not args.use_reid_query:
        print('Warning: use_reid_query is disabled; cross-view ID alignment quality may drop.')

    device = torch.device(args.device)

    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    # Load weights
    if args.resume:
        model = load_model(model, args.resume)
        print(f"Loaded model from {args.resume}")

    # Setup transforms
    transforms = make_inference_transforms()

    # Discover synchronized frame count across all cameras.
    frame_files_by_cam, num_frames = collect_frame_files(args.scene_dir, args.camera_names)
    print(f"Found {num_frames} synchronized frames across {len(args.camera_names)} cameras")
    for cam in args.camera_names:
        print(f"  {cam}: {len(frame_files_by_cam[cam])} frames")

    if num_frames == 0:
        raise ValueError("No synchronized frames found across selected cameras.")

    # Setup video writer
    first_cam = args.camera_names[0]
    first_img_path = osp.join(args.scene_dir, first_cam, 'images', frame_files_by_cam[first_cam][0])
    sample_img = cv2.imread(first_img_path)
    h, w = sample_img.shape[:2]
    # Create a grid layout so many cameras (e.g., 7) remain visible.
    num_cams = len(args.camera_names)
    grid_cols = min(3, num_cams)
    grid_rows = int(math.ceil(num_cams / grid_cols))
    total_w = w * grid_cols
    total_h = h * grid_rows
    requested_output = Path(args.output_video)
    if requested_output.suffix.lower() != '.avi':
        print(f"Requested {args.output_video}; forcing AVI output for compatibility.")
    forced_output = str(requested_output.with_suffix('.avi'))

    out_video, output_video_path, output_codec = create_video_writer(
        forced_output,
        (total_w, total_h),
        fps=float(args.output_fps),
    )
    if output_video_path != forced_output:
        print(f"Requested output {forced_output} not supported, using {output_video_path} ({output_codec}).")
    else:
        print(f"Using video codec {output_codec} for output {output_video_path}.")
    print(f"Output FPS set to {float(args.output_fps):.2f}")

    # Run tracking
    track_instances_list = None
    color_map = {}
    last_results = None

    for frame_idx in tqdm(range(num_frames), desc='Processing frames', unit='frame', dynamic_ncols=True):
        # Load multi-view frames
        pil_images, ori_sizes = load_multiview_frames(
            args.scene_dir, args.camera_names, frame_files_by_cam, frame_idx
        )

        if len(pil_images) != len(args.camera_names):
            print(f"Warning: frame {frame_idx} missing some views, skipping")
            continue

        # Transform images
        transformed_imgs = []
        for img in pil_images:
            # Apply transforms (treating each as a single-frame sequence)
            imgs_t, _ = transforms([img], [{}])
            transformed_imgs.append(imgs_t[0])

        # Run inference
        with torch.no_grad():
            results = model.inference_single_image_multiview(
                imgs=[[img.to(device)] for img in transformed_imgs],
                ori_img_sizes=ori_sizes,
                track_instances_list=track_instances_list,
            )
        last_results = results

        # Update track instances for next frame
        track_instances_list = [
            results['views'][v]['track_instances']
            for v in range(len(args.camera_names))
        ]

        # Visualize
        np_images = [np.array(img)[:, :, ::-1] for img in pil_images]  # RGB -> BGR
        vis_images, color_map = draw_tracks_multiview(
            np_images, results, args.camera_names,
            global_id_mapping=results.get('cross_view_matches'),
            color_map=color_map,
            score_thresh=args.score_thresh,
            box_thickness=args.box_thickness,
            font_scale=args.label_font_scale,
        )

        # Create a fixed-size grid frame.
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        for cam_i, vis_img in enumerate(vis_images):
            if vis_img.shape[0] != h or vis_img.shape[1] != w:
                vis_img = cv2.resize(vis_img, (w, h))
            row = cam_i // grid_cols
            col = cam_i % grid_cols
            y0, y1 = row * h, (row + 1) * h
            x0, x1 = col * w, (col + 1) * w
            canvas[y0:y1, x0:x1] = vis_img
        combined = canvas
        out_video.write(combined)

    out_video.release()
    print(f"Output saved to {output_video_path}")

    # Print cross-view statistics
    global_mapping = {} if last_results is None else last_results.get('cross_view_matches', {})
    print(f"\nCross-view tracking summary:")
    print(f"  Global IDs assigned: {len(global_mapping)}")
    for gid, locals_list in global_mapping.items():
        views_str = ", ".join([f"view{v}:id{lid}" for v, lid in locals_list])
        print(f"  Global ID {gid}: {views_str}")


if __name__ == '__main__':
    main()
