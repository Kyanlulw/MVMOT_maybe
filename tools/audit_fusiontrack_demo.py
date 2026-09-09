"""Run the normal demo while recording predictions before track-ID filtering.

Use the same arguments as demo_multiview.py, plus --audit_dir and --audit_frames.
The audit observes the first N synchronized frames and does not alter predictions.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.multiview.multiview import MultiviewMOTR


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--audit_dir', type=Path, default=Path('inference_audit'))
    parser.add_argument('--audit_frames', type=int, default=3)
    options, remaining = parser.parse_known_args()
    options.audit_dir.mkdir(parents=True, exist_ok=True)
    original = MultiviewMOTR._forward_single_image
    calls = 0
    records = []

    def observed(model, samples, tracks):
        nonlocal calls
        result = original(model, samples, tracks)
        frame, camera = divmod(calls, model.num_cams)
        calls += 1
        if frame >= options.audit_frames:
            return result
        tensor = samples.tensors[0].detach().cpu()
        mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
        std = tensor.new_tensor([.229, .224, .225])[:, None, None]
        pixels = ((tensor*std+mean).clamp(0, 1)*255).byte().permute(1, 2, 0).numpy()
        base = Image.fromarray(pixels)
        width, height = base.size
        for stage, prefix in [('detection', 'det_'), ('tracking', '')]:
            if prefix+'pred_logits' not in result:
                continue
            scores = result[prefix+'pred_logits'][0, :, 0].sigmoid().detach().cpu()
            boxes = result[prefix+'pred_boxes'][0].detach().cpu()
            record = dict(frame_index=frame, camera_index=camera, stage=stage,
                          queries=len(scores), max_score=float(scores.max()),
                          counts={str(t): int((scores >= t).sum()) for t in (.1, .3, .4, .5, .7)},
                          scores=scores.tolist(), boxes_cxcywh=boxes.tolist())
            records.append(record)
            im = base.copy()
            draw = ImageDraw.Draw(im)
            for index in scores.argsort(descending=True)[:100].tolist():
                if scores[index] < .3:
                    continue
                cx, cy, bw, bh = boxes[index].tolist()
                xyxy = ((cx-bw/2)*width, (cy-bh/2)*height,
                        (cx+bw/2)*width, (cy+bh/2)*height)
                draw.rectangle(xyxy, outline='lime', width=2)
                draw.text((max(0, xyxy[0]), max(20, xyxy[1])),
                          f'q{index} {scores[index]:.2f}', fill='yellow')
            draw.rectangle((0, 0, width, 20), fill='black')
            draw.text((3, 3), f'{stage}: before IDs | score >= .3 | top 100', fill='white')
            im.save(options.audit_dir / f'frame{frame:04d}_cam{camera}_{stage}.jpg')
            print(f'AUDIT frame={frame} cam={camera} {stage}: {record["counts"]}')
        (options.audit_dir / 'raw_predictions.json').write_text(json.dumps(records, indent=2))
        return result

    MultiviewMOTR._forward_single_image = observed
    sys.argv = [str(ROOT / 'demo_multiview.py')] + remaining
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        MultiviewMOTR._forward_single_image = original


if __name__ == '__main__':
    main()
