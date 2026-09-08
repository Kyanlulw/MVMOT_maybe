# ------------------------------------------------------------------------
# Multi-View MOT Dataset
# ------------------------------------------------------------------------
# Dataset class for loading synchronized multi-view tracking data.
# Supports multiple overlapping cameras with shared global track IDs.
# ------------------------------------------------------------------------
# 
# Expected data structure:
# 
# data_root/
#   ├── scene_001/
#   │   ├── camera_0/
#   │   │   ├── images/
#   │   │   │   ├── 000001.jpg
#   │   │   │   ├── 000002.jpg
#   │   │   │   └── ...
#   │   │   ├── labels_with_ids/
#   │   │   │   ├── 000001.txt
#   │   │   │   ├── 000002.txt
#   │   │   │   └── ...
#   │   │   └── calibration.txt  (optional)
#   │   ├── camera_1/
#   │   │   ├── images/
#   │   │   ├── labels_with_ids/
#   │   │   └── calibration.txt
#   │   └── ...
#   └── scene_002/
#       └── ...
# 
# Label format (per line):  class_id  track_id  cx  cy  w  h
# Calibration file format (optional):
#   Line 1: 3x3 intrinsic matrix (9 values)
#   Line 2: 3x4 extrinsic matrix (12 values)
# 
# Data text file format:
#   scene_name,camera_name,frame_number
#   e.g.: scene_001,camera_0,000001
# ------------------------------------------------------------------------

import os
import os.path as osp
import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image

import datasets.transforms as T
from models.structures import Instances


class MultiViewMOTDetection:
    """
    Multi-view MOT dataset that loads synchronized frames from
    multiple cameras and returns per-view images with annotations.
    
    Global track IDs are shared across views: the same physical object
    seen from multiple cameras has the same track ID in all views.
    """

    def __init__(self, args, data_txt_path: str, seqs_folder: str, transforms):
        self.args = args
        self._transforms = transforms
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = getattr(args, 'vis', False)
        self.num_views = getattr(args, 'num_cams', getattr(args, 'num_views', 2))
        self.seqs_folder = seqs_folder

        # Parse data text file to build multi-view sample index
        self.scenes = []  # List of scene configs
        self.frame_index = []  # List of (scene_idx, frame_start_idx)
        self.video_dict = {}
        self.scene_frame_offsets = []  # Global frame offset per scene
        self._next_global_frame_offset = 0

        self._parse_data_file(data_txt_path, seqs_folder)
        for scene in self.scenes:
            available_views = len(scene['cameras'])
            if available_views < self.num_views:
                raise ValueError(
                    f"Scene {scene['name']} provides {available_views} cameras, "
                    f"but --num_cams={self.num_views}."
                )
            if available_views > self.num_views:
                # A manifest generated for all WildTrack cameras can be reused
                # for a lower-view experiment. Preserve the manifest order so
                # camera selection stays deterministic across train and val.
                scene['cameras'] = scene['cameras'][:self.num_views]
                print(
                    f"MultiView: using the first {self.num_views} of "
                    f"{available_views} cameras for scene {scene['name']}"
                )

        # Video sampler (same logic as single-view)
        self.sampler_steps = args.sampler_steps
        self.lengths = args.sampler_lengths
        print(f"MultiView: sampler_steps={self.sampler_steps} lengths={self.lengths}")
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

    def _parse_data_file(self, data_txt_path: str, seqs_folder: str):
        """
        Parse the multi-view data file.
        
        Supports two formats:
        1. JSON format: structured multi-view scene descriptions
        2. Text format: simple listing of scene/camera/frame entries
        """
        # Auto-detect JSON payloads even when files use custom extensions
        # such as ".train" / ".val".
        is_json = data_txt_path.endswith('.json')
        if not is_json:
            try:
                with open(data_txt_path, 'r') as f:
                    for raw_line in f:
                        stripped = raw_line.strip()
                        if stripped:
                            is_json = stripped[0] in ('{', '[')
                            break
            except OSError:
                is_json = False

        if is_json:
            self._parse_json_format(data_txt_path, seqs_folder)
        else:
            self._parse_text_format(data_txt_path, seqs_folder)

    def _parse_json_format(self, json_path: str, seqs_folder: str):
        """
        Parse JSON format multi-view data file.
        
        Expected JSON structure:
        {
            "scenes": [
                {
                    "name": "scene_001",
                    "cameras": ["camera_0", "camera_1"],
                    "num_frames": 100,
                    "calibration": {  // optional
                        "homographies": {
                            "0_1": [3x3 matrix as list of lists]
                        }
                    }
                }
            ]
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        for scene_cfg in data['scenes']:
            scene_name = scene_cfg['name']
            cameras = scene_cfg['cameras']
            requested_num_frames = int(scene_cfg.get('num_frames', 0))
            start_frame = int(scene_cfg.get('start_frame', 0))
            frame_stride = int(scene_cfg.get('frame_stride', 1))
            if frame_stride <= 0:
                raise ValueError(f"frame_stride must be > 0, got {frame_stride} for scene {scene_name}")

            scene = {
                'name': scene_name,
                'cameras': cameras,
                'num_frames': 0,
                'calibration': scene_cfg.get('calibration', None),
                'img_paths': {},
                'label_paths': {},
            }

            selected_lengths = []

            for cam in cameras:
                cam_img_dir = osp.join(seqs_folder, scene_name, cam, 'images')
                cam_label_dir = osp.join(seqs_folder, scene_name, cam, 'labels_with_ids')

                img_files = sorted([
                    osp.join(cam_img_dir, f) for f in os.listdir(cam_img_dir)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ]) if osp.isdir(cam_img_dir) else []

                img_files = img_files[start_frame::frame_stride]
                if requested_num_frames > 0:
                    img_files = img_files[:requested_num_frames]

                label_files = [
                    f.replace('images', 'labels_with_ids')
                     .replace('.jpg', '.txt')
                     .replace('.png', '.txt')
                     .replace('.jpeg', '.txt')
                    for f in img_files
                ]

                scene['img_paths'][cam] = img_files
                scene['label_paths'][cam] = label_files
                selected_lengths.append(len(img_files))

            if len(selected_lengths) == 0:
                continue

            scene['num_frames'] = min(selected_lengths)

            for cam in cameras:
                scene['img_paths'][cam] = scene['img_paths'][cam][:scene['num_frames']]
                scene['label_paths'][cam] = scene['label_paths'][cam][:scene['num_frames']]

            if scene['num_frames'] <= 0:
                continue

            scene_idx = len(self.scenes)
            self.scenes.append(scene)
            self.scene_frame_offsets.append(self._next_global_frame_offset)
            self._next_global_frame_offset += scene['num_frames']

            # Register video
            video_name = osp.join(seqs_folder, scene_name)
            if video_name not in self.video_dict:
                self.video_dict[video_name] = len(self.video_dict)

            # Create frame indices
            max_start = scene['num_frames'] - (self.num_frames_per_batch - 1) * self.sample_interval
            for frame_idx in range(max_start):
                self.frame_index.append((scene_idx, frame_idx))

    def _parse_text_format(self, txt_path: str, seqs_folder: str):
        """
        Parse text format multi-view data file.
        
        Each line: scene_name/camera_name/images/frame.jpg
        
        Groups lines by scene to find multi-view setups.
        """
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Group by scene
        scene_cameras = {}
        for line in lines:
            # Skip obvious JSON/object syntax lines when a JSON-like file was
            # accidentally routed here (e.g., wrong extension).
            if line in {'{', '}', '[', ']', ','}:
                continue
            if ':' in line and '"' in line and ',' not in line and '/' not in line:
                continue

            parts = line.split(',') if ',' in line else line.split('/')
            if len(parts) >= 2:
                scene_name = parts[0].strip()
                if scene_name not in scene_cameras:
                    scene_cameras[scene_name] = {}

                img_path = osp.join(seqs_folder, line.split(',')[0].strip()) if ',' in line else osp.join(seqs_folder, line)

                # Infer camera name from path structure
                path_parts = img_path.replace('\\', '/').split('/')
                for i, p in enumerate(path_parts):
                    if p.startswith('camera') or p.startswith('cam') or p.startswith('view'):
                        cam_name = p
                        break
                else:
                    cam_name = 'camera_0'

                if cam_name not in scene_cameras[scene_name]:
                    scene_cameras[scene_name][cam_name] = []
                scene_cameras[scene_name][cam_name].append(img_path)

        # Build scenes
        for scene_name, cameras in scene_cameras.items():
            camera_names = sorted(cameras.keys())
            min_frames = min(len(cameras[c]) for c in camera_names)

            scene = {
                'name': scene_name,
                'cameras': camera_names[:self.num_views],
                'num_frames': min_frames,
                'calibration': None,
                'img_paths': {},
                'label_paths': {},
            }

            for cam in camera_names[:self.num_views]:
                img_files = sorted(cameras[cam])[:min_frames]
                label_files = [
                    f.replace('images', 'labels_with_ids')
                     .replace('.jpg', '.txt')
                     .replace('.png', '.txt')
                    for f in img_files
                ]
                scene['img_paths'][cam] = img_files
                scene['label_paths'][cam] = label_files

            scene_idx = len(self.scenes)
            self.scenes.append(scene)
            self.scene_frame_offsets.append(self._next_global_frame_offset)
            self._next_global_frame_offset += scene['num_frames']

            video_name = osp.join(seqs_folder, scene_name)
            if video_name not in self.video_dict:
                self.video_dict[video_name] = len(self.video_dict)

            max_start = min_frames - (self.num_frames_per_batch - 1) * self.sample_interval
            for frame_idx in range(max(1, max_start)):
                self.frame_index.append((scene_idx, frame_idx))

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            return
        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print(f"MultiView Dataset: epoch {epoch} period_idx={self.period_idx}")
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        print(f"MultiView Dataset: epoch {self.current_epoch} finishes")
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _load_single_frame(self, scene_idx: int, cam_name: str, frame_idx: int):
        """Load a single frame from a specific camera view."""
        scene = self.scenes[scene_idx]
        img_path = scene['img_paths'][cam_name][frame_idx]
        label_path = scene['label_paths'][cam_name][frame_idx]

        img = Image.open(img_path)
        w, h = img._size
        assert w > 0 and h > 0, f"Invalid image {img_path} with shape {w} {h}"

        targets = {}
        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            labels = labels0.copy()
            # Convert normalized cxcywh to pixel xyxy
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            labels = np.zeros((0, 6), dtype=np.float32)

        video_name = osp.join(self.seqs_folder, scene['name'])
        obj_idx_offset = self.video_dict[video_name] * 100000

        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(frame_idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        targets['view_id'] = cam_name

        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            # Global object ID: shared across views for the same physical object
            track_id = int(label[1])
            obj_id = track_id + obj_idx_offset if track_id >= 0 else track_id
            targets['obj_ids'].append(int(obj_id))

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.long)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        if len(targets['boxes']) > 0:
            targets['boxes'][:, 0::2].clamp_(min=0, max=w)
            targets['boxes'][:, 1::2].clamp_(min=0, max=h)

        return img, targets

    def _get_sample_range(self, start_idx: int):
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        else:
            raise ValueError(f'Invalid sample mode: {self.sample_mode}')

        end_idx = start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1
        return start_idx, end_idx, sample_interval

    def __getitem__(self, idx):
        scene_idx, base_frame_idx = self.frame_index[idx]
        scene = self.scenes[scene_idx]
        cameras = scene['cameras']

        sample_start, sample_end, sample_interval = self._get_sample_range(base_frame_idx)
        # Clamp to available frames
        sample_end = min(sample_end, scene['num_frames'])
        sampled_frame_indices = list(range(sample_start, sample_end, sample_interval))
        scene_global_offset = self.scene_frame_offsets[scene_idx]
        global_frame_idxs = [scene_global_offset + frame_idx for frame_idx in sampled_frame_indices]

        data = {}

        # Load frames for all views and all time steps
        imgs_multiview = []  # List[num_views] of List[num_frames] of Tensor
        gt_instances_multiview = []  # List[num_views] of List[num_frames] of Instances

        for cam_idx, cam_name in enumerate(cameras):
            images_for_view = []
            targets_for_view = []

            for frame_idx in sampled_frame_indices:
                img, targets = self._load_single_frame(scene_idx, cam_name, frame_idx)
                images_for_view.append(img)
                targets_for_view.append(targets)

            # Apply transforms per view
            if self._transforms is not None:
                images_for_view, targets_for_view = self._transforms(
                    images_for_view, targets_for_view
                )

            gt_instances_for_view = []
            for img_i, targets_i in zip(images_for_view, targets_for_view):
                gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
                gt_instances_for_view.append(gt_instances_i)

            imgs_multiview.append(images_for_view)
            gt_instances_multiview.append(gt_instances_for_view)

        data['imgs_multiview'] = imgs_multiview
        data['gt_instances_multiview'] = gt_instances_multiview

        # Canonical keys consumed by MultiviewMOTR.forward.
        data['imgs'] = imgs_multiview
        data['gt_instances'] = gt_instances_multiview
        data['global_frame_idxs'] = global_frame_idxs

        # Keep explicit single-view aliases for optional debugging/visualization.
        data['imgs_single_view'] = imgs_multiview[0]
        data['gt_instances_single_view'] = gt_instances_multiview[0]

        # Calibration data (if available)
        if scene.get('calibration') is not None:
            data['calibration'] = scene['calibration']

        if self.vis:
            data['ori_img'] = []
            for v in range(len(cameras)):
                view_ori = []
                for frame_idx in sampled_frame_indices:
                    img, _ = self._load_single_frame(scene_idx, cameras[v], frame_idx)
                    view_ori.append(np.array(img))
                data['ori_img'].append(view_ori)

        return data

    def __len__(self):
        return len(self.frame_index)


def make_multiview_transforms(image_set, args=None):
    """Create transforms for multi-view MOT dataset."""
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if getattr(args, 'smoke_train', False):
        scales = [256]
        max_size = 320
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333

    if image_set == 'train':
        color_transforms = []
        scale_transforms = [
            T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=max_size),
            normalize,
        ]
        return T.MotCompose(color_transforms + scale_transforms)

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([256 if getattr(args, 'smoke_train', False) else 800],
                              max_size=320 if getattr(args, 'smoke_train', False) else 1333),
            normalize,
        ])

    raise ValueError(f'Unknown image_set: {image_set}')


def build(image_set, args):
    """Build multi-view MOT dataset."""
    root = Path(args.mot_path)
    assert root.exists(), f'Provided MOT path {root} does not exist'

    transforms = make_multiview_transforms(image_set, args)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
    else:
        data_txt_path = args.data_txt_path_val

    dataset = MultiViewMOTDetection(
        args,
        data_txt_path=data_txt_path,
        seqs_folder=str(root),
        transforms=transforms,
    )
    return dataset
