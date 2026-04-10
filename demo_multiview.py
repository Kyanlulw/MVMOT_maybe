# ------------------------------------------------------------------------
# Multi-View MOTR Demo
# ------------------------------------------------------------------------
# Inference script for multi-view multi-object tracking with overlapping
# cameras. Processes synchronized frames from multiple camera views and
# outputs per-view tracks with cross-view global ID association.
# ------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

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


def load_multiview_frames(scene_dir, camera_names, frame_idx):
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
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
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

    output_images = []
    for v, cam_name in enumerate(camera_names):
        img = images[v].copy()
        view_result = results['views'][v]
        track_instances = view_result['track_instances']

        if track_instances is not None and len(track_instances) > 0:
            boxes = track_instances.boxes.cpu().numpy() if hasattr(track_instances, 'boxes') else []
            scores = track_instances.scores.cpu().numpy() if hasattr(track_instances, 'scores') else []
            obj_ids = track_instances.obj_idxes.cpu().numpy() if hasattr(track_instances, 'obj_idxes') else []

            for i in range(len(boxes)):
                if scores[i] < 0.5:
                    continue

                obj_id = int(obj_ids[i])
                if obj_id < 0:
                    continue

                # Get global ID if available
                global_id = -1
                if global_id_mapping:
                    for gid, locals_list in global_id_mapping.items():
                        for vid, lid in locals_list:
                            if vid == v and lid == obj_id:
                                global_id = gid
                                break

                # Use global ID for color if available, else local ID
                display_id = global_id if global_id >= 0 else obj_id
                if display_id not in color_map:
                    color_map[display_id] = tuple(
                        int(c) for c in np.random.randint(0, 255, 3)
                    )
                color = color_map[display_id]

                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label with both local and global ID
                if global_id >= 0:
                    label = f"G{global_id}|L{obj_id}"
                else:
                    label = f"L{obj_id}"
                label += f" {scores[i]:.2f}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add camera label
        cv2.putText(img, cam_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        output_images.append(img)

    return output_images, color_map


def main():
    parser = argparse.ArgumentParser('Multi-View MOTR Demo', parents=[get_args_parser()])
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Path to the scene directory with camera subdirectories')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['camera_0', 'camera_1'],
                       help='Names of camera subdirectories')
    parser.add_argument('--output_video', type=str, default='multiview_output.mp4',
                       help='Output video path')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                       help='Score threshold for visualization')
    args = parser.parse_args()

    # Override some args for demo
    args.meta_arch = 'multiview_motr'
    args.dataset_file = 'e2e_mv_mot'
    args.num_views = len(args.camera_names)

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

    # Discover frames
    first_cam = args.camera_names[0]
    img_dir = osp.join(args.scene_dir, first_cam, 'images')
    frame_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    num_frames = len(frame_files)
    print(f"Found {num_frames} frames across {len(args.camera_names)} cameras")

    # Setup video writer
    sample_img = cv2.imread(osp.join(img_dir, frame_files[0]))
    h, w = sample_img.shape[:2]
    # Create side-by-side layout
    total_w = w * len(args.camera_names)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_video, fourcc, 30, (total_w, h))

    # Run tracking
    track_instances_list = None
    color_map = {}

    for frame_idx in range(num_frames):
        # Load multi-view frames
        pil_images, ori_sizes = load_multiview_frames(
            args.scene_dir, args.camera_names, frame_idx
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
                imgs=[img.unsqueeze(0).to(device) for img in transformed_imgs],
                ori_img_sizes=ori_sizes,
                track_instances_list=track_instances_list,
            )

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
        )

        # Create side-by-side frame
        # Resize all to same height
        resized = []
        for vis_img in vis_images:
            if vis_img.shape[0] != h or vis_img.shape[1] != w:
                vis_img = cv2.resize(vis_img, (w, h))
            resized.append(vis_img)
        combined = np.concatenate(resized, axis=1)
        out_video.write(combined)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{num_frames}")

    out_video.release()
    print(f"Output saved to {args.output_video}")

    # Print cross-view statistics
    global_mapping = results.get('cross_view_matches', {})
    print(f"\nCross-view tracking summary:")
    print(f"  Global IDs assigned: {len(global_mapping)}")
    for gid, locals_list in global_mapping.items():
        views_str = ", ".join([f"view{v}:id{lid}" for v, lid in locals_list])
        print(f"  Global ID {gid}: {views_str}")


if __name__ == '__main__':
    main()
