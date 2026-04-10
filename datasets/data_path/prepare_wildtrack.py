"""
Prepare Wildtrack for Multi-View MOTR.

This script converts raw Wildtrack into the folder/label format expected by
`datasets/multiview_mot.py`:

output_root/
  scene_name/
    camera_0/
      images/
      labels_with_ids/
    camera_1/
      images/
      labels_with_ids/
    ...

Label format per line:
  class_id track_id cx cy w h
where cx, cy, w, h are normalized to [0, 1].

It also writes split files under datasets/data_path:
  multiview_wildtrack.train
  multiview_wildtrack.val

JSON split files use `start_frame` + `num_frames`, so train/val can share one
converted scene without data duplication.
"""

import argparse
import json
import os
import os.path as osp
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str):
    if osp.exists(dst):
        return
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _find_dir(root: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        p = osp.join(root, name)
        if osp.isdir(p):
            return p
    return None


def _first_key(d: Dict, keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _resolve_view_index(raw_view_num: int, num_views: int) -> Optional[int]:
    # Support both 0-based and 1-based camera indices.
    if 0 <= raw_view_num < num_views:
        return raw_view_num
    if 1 <= raw_view_num <= num_views:
        return raw_view_num - 1
    return None


def _extract_bbox(view_item: Dict) -> Optional[Tuple[float, float, float, float]]:
    x1 = _first_key(view_item, ["xmin", "x1", "left"])
    y1 = _first_key(view_item, ["ymin", "y1", "top"])
    x2 = _first_key(view_item, ["xmax", "x2", "right"])
    y2 = _first_key(view_item, ["ymax", "y2", "bottom"])

    if None in (x1, y1, x2, y2):
        bbox = view_item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            return None

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _find_image_for_frame(img_dir: str, frame_stem: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = osp.join(img_dir, frame_stem + ext)
        if osp.isfile(p):
            return p
    # fallback: allow zero-padded mismatch by integer value
    try:
        frame_id = int(frame_stem)
    except ValueError:
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        for width in (6, 7, 8):
            p = osp.join(img_dir, f"{frame_id:0{width}d}{ext}")
            if osp.isfile(p):
                return p
    return None


def convert_wildtrack(
    wildtrack_root: str,
    output_root: str,
    scene_name: str,
    num_views: int,
    train_ratio: float,
    split_output_dir: str,
    train_name: str,
    val_name: str,
):
    annotations_dir = _find_dir(
        wildtrack_root,
        ["annotations_positions", "annotations", osp.join("annotations", "positions")],
    )
    image_subsets_dir = _find_dir(
        wildtrack_root,
        ["Image_subsets", "image_subsets", "images"],
    )

    if annotations_dir is None:
        raise FileNotFoundError("Could not find Wildtrack annotations directory")
    if image_subsets_dir is None:
        raise FileNotFoundError("Could not find Wildtrack image subsets directory")

    out_scene_dir = osp.join(output_root, scene_name)
    camera_dirs = []
    for cam_idx in range(num_views):
        cam_name = f"camera_{cam_idx}"
        cam_dir = osp.join(out_scene_dir, cam_name)
        img_dir = osp.join(cam_dir, "images")
        label_dir = osp.join(cam_dir, "labels_with_ids")
        _safe_mkdir(img_dir)
        _safe_mkdir(label_dir)
        camera_dirs.append((img_dir, label_dir))

    # Map Wildtrack cameras C1..Cn to camera_0..camera_{n-1}
    source_cam_dirs = []
    for cam_idx in range(num_views):
        candidates = [
            osp.join(image_subsets_dir, f"C{cam_idx + 1}"),
            osp.join(image_subsets_dir, f"c{cam_idx + 1}"),
            osp.join(image_subsets_dir, str(cam_idx + 1)),
        ]
        found = None
        for p in candidates:
            if osp.isdir(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Could not find camera folder for view {cam_idx} under {image_subsets_dir}")
        source_cam_dirs.append(found)

    ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])
    if len(ann_files) == 0:
        raise RuntimeError("No annotation json files found for Wildtrack")

    converted_frames = 0
    for ann_name in ann_files:
        frame_stem = osp.splitext(ann_name)[0]
        ann_path = osp.join(annotations_dir, ann_name)

        with open(ann_path, 'r') as f:
            frame_ann = json.load(f)

        # Prepare images and empty label files for each camera first.
        image_sizes = []
        copied_any_view = False
        for cam_idx in range(num_views):
            src_img = _find_image_for_frame(source_cam_dirs[cam_idx], frame_stem)
            if src_img is None:
                image_sizes.append(None)
                continue

            dst_img = osp.join(camera_dirs[cam_idx][0], osp.basename(src_img))
            _link_or_copy(src_img, dst_img)
            with Image.open(src_img) as im:
                image_sizes.append(im.size)  # (w, h)
            copied_any_view = True

            dst_label = osp.join(camera_dirs[cam_idx][1], frame_stem + ".txt")
            if not osp.isfile(dst_label):
                open(dst_label, 'w').close()

        if not copied_any_view:
            continue

        # Collect per-camera label lines.
        per_cam_lines = [[] for _ in range(num_views)]

        if isinstance(frame_ann, dict):
            # Some variants wrap list in a dict key.
            if 'annotations' in frame_ann and isinstance(frame_ann['annotations'], list):
                frame_ann = frame_ann['annotations']
            elif 'persons' in frame_ann and isinstance(frame_ann['persons'], list):
                frame_ann = frame_ann['persons']
            else:
                frame_ann = []

        for person in frame_ann:
            if not isinstance(person, dict):
                continue
            person_id = _first_key(person, ["personID", "track_id", "id", "person_id"])
            if person_id is None:
                continue
            try:
                person_id = int(person_id)
            except Exception:
                continue

            views = person.get("views", person.get("bboxes", []))
            if not isinstance(views, list):
                continue

            for v in views:
                if not isinstance(v, dict):
                    continue
                raw_view_num = _first_key(v, ["viewNum", "view", "cameraID", "cam_id"])
                if raw_view_num is None:
                    continue
                try:
                    raw_view_num = int(raw_view_num)
                except Exception:
                    continue

                cam_idx = _resolve_view_index(raw_view_num, num_views)
                if cam_idx is None:
                    continue

                if image_sizes[cam_idx] is None:
                    continue
                img_w, img_h = image_sizes[cam_idx]

                bbox = _extract_bbox(v)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox

                # Clip bbox to image area.
                x1 = max(0.0, min(x1, img_w - 1.0))
                y1 = max(0.0, min(y1, img_h - 1.0))
                x2 = max(0.0, min(x2, img_w - 1.0))
                y2 = max(0.0, min(y2, img_h - 1.0))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = ((x1 + x2) * 0.5) / img_w
                cy = ((y1 + y2) * 0.5) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                if bw <= 0 or bh <= 0:
                    continue

                per_cam_lines[cam_idx].append(
                    f"0 {person_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                )

        # Save labels.
        for cam_idx in range(num_views):
            if image_sizes[cam_idx] is None:
                continue
            dst_label = osp.join(camera_dirs[cam_idx][1], frame_stem + ".txt")
            with open(dst_label, 'w') as f:
                f.writelines(per_cam_lines[cam_idx])

        converted_frames += 1

    if converted_frames <= 1:
        raise RuntimeError("Too few frames converted from Wildtrack")

    train_frames = int(converted_frames * train_ratio)
    train_frames = max(1, min(train_frames, converted_frames - 1))
    val_frames = converted_frames - train_frames

    cameras = [f"camera_{i}" for i in range(num_views)]
    train_json = {
        "scenes": [
            {
                "name": scene_name,
                "cameras": cameras,
                "start_frame": 0,
                "num_frames": train_frames,
            }
        ]
    }
    val_json = {
        "scenes": [
            {
                "name": scene_name,
                "cameras": cameras,
                "start_frame": train_frames,
                "num_frames": val_frames,
            }
        ]
    }

    _safe_mkdir(split_output_dir)
    train_path = osp.join(split_output_dir, train_name)
    val_path = osp.join(split_output_dir, val_name)

    with open(train_path, 'w') as f:
        json.dump(train_json, f, indent=4)
    with open(val_path, 'w') as f:
        json.dump(val_json, f, indent=4)

    print("=" * 60)
    print("Wildtrack conversion finished")
    print(f"Converted frames: {converted_frames}")
    print(f"Output scene dir: {out_scene_dir}")
    print(f"Train split: {train_path} ({train_frames} frames)")
    print(f"Val split:   {val_path} ({val_frames} frames)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert Wildtrack to Multi-View MOTR format")
    parser.add_argument('--wildtrack_root', type=str, required=True,
                        help='Path to raw Wildtrack root')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output root used as --mot_path for training')
    parser.add_argument('--scene_name', type=str, default='wildtrack_scene',
                        help='Scene folder name to create under output_root')
    parser.add_argument('--num_views', type=int, default=7,
                        help='Number of Wildtrack camera views to use')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train split ratio by frame count')
    parser.add_argument('--split_output_dir', type=str, default='./datasets/data_path',
                        help='Where to save generated train/val JSON split files')
    parser.add_argument('--train_split_name', type=str, default='multiview_wildtrack.train',
                        help='Train split JSON filename')
    parser.add_argument('--val_split_name', type=str, default='multiview_wildtrack.val',
                        help='Val split JSON filename')

    args = parser.parse_args()

    convert_wildtrack(
        wildtrack_root=args.wildtrack_root,
        output_root=args.output_root,
        scene_name=args.scene_name,
        num_views=args.num_views,
        train_ratio=args.train_ratio,
        split_output_dir=args.split_output_dir,
        train_name=args.train_split_name,
        val_name=args.val_split_name,
    )


if __name__ == '__main__':
    main()
