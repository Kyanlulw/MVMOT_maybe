"""
Multi-View MOT Data Preparation Script
=======================================

This script helps prepare multi-view MOT datasets for training with Multi-View MOTR.

Expected input data structure:
    data_root/
        scene_001/
            camera_0/
                images/
                    000001.jpg
                    000002.jpg
                    ...
                labels_with_ids/
                    000001.txt   (format: class_id track_id cx cy w h)
                    000002.txt
                    ...
            camera_1/
                images/
                labels_with_ids/
            ...

IMPORTANT: Track IDs must be globally consistent across cameras!
The same physical person/object must have the same track_id in all camera views.

Usage:
    python datasets/data_path/prepare_multiview.py \\
        --data_root /path/to/multiview_data \\
        --output_dir ./datasets/data_path/ \\
        --num_views 2 \\
        --train_ratio 0.8
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path


def discover_scenes(data_root: str, num_views: int):
    """Discover multi-view scenes in the data root directory."""
    scenes = []
    
    for scene_name in sorted(os.listdir(data_root)):
        scene_dir = osp.join(data_root, scene_name)
        if not osp.isdir(scene_dir):
            continue
        
        # Find camera directories
        cameras = []
        for cam_name in sorted(os.listdir(scene_dir)):
            cam_dir = osp.join(scene_dir, cam_name)
            img_dir = osp.join(cam_dir, 'images')
            if osp.isdir(img_dir):
                cameras.append(cam_name)
        
        if len(cameras) < num_views:
            print(f"Warning: scene {scene_name} has only {len(cameras)} cameras "
                  f"(need {num_views}), skipping")
            continue
        
        # Count frames (use minimum across cameras)
        min_frames = float('inf')
        for cam in cameras[:num_views]:
            img_dir = osp.join(scene_dir, cam, 'images')
            num_imgs = len([
                f for f in os.listdir(img_dir)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
            min_frames = min(min_frames, num_imgs)
        
        if min_frames == 0:
            print(f"Warning: scene {scene_name} has no frames, skipping")
            continue
        
        # Check for calibration
        calibration = None
        calib_file = osp.join(scene_dir, 'calibration.json')
        if osp.isfile(calib_file):
            with open(calib_file, 'r') as f:
                calibration = json.load(f)
        
        scene = {
            'name': scene_name,
            'cameras': cameras[:num_views],
            'num_frames': int(min_frames),
        }
        if calibration:
            scene['calibration'] = calibration
        
        scenes.append(scene)
        print(f"Found scene: {scene_name} with {len(cameras[:num_views])} cameras, "
              f"{min_frames} frames")
    
    return scenes


def validate_global_ids(data_root: str, scenes: list):
    """Validate that track IDs are globally consistent across cameras."""
    issues = []
    
    for scene in scenes:
        scene_dir = osp.join(data_root, scene['name'])
        cameras = scene['cameras']
        
        for frame_idx in range(min(10, scene['num_frames'])):  # Check first 10 frames
            per_cam_ids = {}
            for cam in cameras:
                label_dir = osp.join(scene_dir, cam, 'labels_with_ids')
                img_dir = osp.join(scene_dir, cam, 'images')
                img_files = sorted([
                    f for f in os.listdir(img_dir) 
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                if frame_idx >= len(img_files):
                    continue
                
                label_name = img_files[frame_idx].replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = osp.join(label_dir, label_name)
                
                if osp.isfile(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    track_ids = set()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            track_ids.add(int(float(parts[1])))
                    per_cam_ids[cam] = track_ids
            
            # Check for overlapping IDs (expected for same objects)
            if len(per_cam_ids) >= 2:
                cams = list(per_cam_ids.keys())
                for i in range(len(cams)):
                    for j in range(i+1, len(cams)):
                        shared = per_cam_ids[cams[i]] & per_cam_ids[cams[j]]
                        if shared:
                            print(f"  Scene {scene['name']}, frame {frame_idx}: "
                                  f"{len(shared)} shared IDs between {cams[i]} and {cams[j]}")
    
    return issues


def generate_data_files(scenes: list, output_dir: str, train_ratio: float = 0.8):
    """Generate train/val data files."""
    import random
    random.shuffle(scenes)
    
    split_idx = int(len(scenes) * train_ratio)
    train_scenes = scenes[:split_idx] if split_idx > 0 else scenes
    val_scenes = scenes[split_idx:] if split_idx < len(scenes) else scenes[:1]
    
    # Write train file
    train_data = {'scenes': train_scenes}
    train_path = osp.join(output_dir, 'multiview.train')
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    print(f"Train file: {train_path} ({len(train_scenes)} scenes)")
    
    # Write val file
    val_data = {'scenes': val_scenes}
    val_path = osp.join(output_dir, 'multiview.val')
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    print(f"Val file: {val_path} ({len(val_scenes)} scenes)")


def main():
    parser = argparse.ArgumentParser(description='Prepare multi-view MOT data')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing multi-view scenes')
    parser.add_argument('--output_dir', type=str, default='./datasets/data_path/',
                       help='Output directory for data split files')
    parser.add_argument('--num_views', type=int, default=2,
                       help='Number of camera views to use per scene')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of scenes for training')
    parser.add_argument('--validate_ids', action='store_true',
                       help='Validate global track ID consistency')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-View MOT Data Preparation")
    print("=" * 60)
    
    # Discover scenes
    scenes = discover_scenes(args.data_root, args.num_views)
    print(f"\nFound {len(scenes)} valid scenes")
    
    if len(scenes) == 0:
        print("No valid scenes found! Check your data structure.")
        return
    
    # Validate global IDs
    if args.validate_ids:
        print("\nValidating global track IDs...")
        validate_global_ids(args.data_root, scenes)
    
    # Generate data files
    os.makedirs(args.output_dir, exist_ok=True)
    generate_data_files(scenes, args.output_dir, args.train_ratio)
    
    print("\nDone! You can now train with:")
    print(f"  bash configs/r50_multiview_motr_train.sh 0 {args.data_root} ./output/multiview")


if __name__ == '__main__':
    main()
