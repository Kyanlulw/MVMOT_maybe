"""Overlay converted training labels on original images; no GPU required."""
import argparse
import colorsys
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene_dir', type=Path, required=True)
    parser.add_argument('--camera_names', nargs='+', required=True)
    parser.add_argument('--frame_stems', nargs='+', help='Exact image stems, without extension')
    parser.add_argument('--output_dir', type=Path, default=Path('gt_overlays'))
    args = parser.parse_args()
    indexed = {}
    for cam in args.camera_names:
        folder = args.scene_dir / cam / 'images'
        files = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                if path.stem in files:
                    raise ValueError(f'Duplicate image stem: {path}')
                files[path.stem] = path
        indexed[cam] = files
    common = sorted(set.intersection(*(set(v) for v in indexed.values())))
    if not common:
        raise ValueError('No matching frame stems across cameras')
    stems = args.frame_stems or list(dict.fromkeys(common[i] for i in (0, len(common)//2, len(common)-1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for stem in stems:
        panels = []
        for cam in args.camera_names:
            image_path = indexed[cam][stem]
            label_path = args.scene_dir / cam / 'labels_with_ids' / f'{stem}.txt'
            im = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(im)
            width, height = im.size
            issues, count, ids = [], 0, set()
            if not label_path.is_file():
                issues.append('MISSING LABEL FILE (training loader treats this as empty)')
                lines = []
            else:
                lines = label_path.read_text().splitlines()
                if not any(line.strip() for line in lines):
                    issues.append('EMPTY LABEL FILE')
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    cls, identity, cx, cy, bw, bh = map(float, line.split())
                    if not all(math.isfinite(x) for x in (cls, identity, cx, cy, bw, bh)):
                        raise ValueError('non-finite value')
                except ValueError as exc:
                    issues.append(f'line {line_no}: invalid six-column label: {exc}')
                    continue
                if cls != 0:
                    issues.append(f'line {line_no}: unexpected pedestrian class {cls}')
                if identity in ids:
                    issues.append(f'line {line_no}: duplicate ID {identity}')
                ids.add(identity)
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    issues.append(f'line {line_no}: invalid normalized center/size')
                if bw <= 0 or bh <= 0:
                    continue
                box = ((cx-bw/2)*width, (cy-bh/2)*height,
                       (cx+bw/2)*width, (cy+bh/2)*height)
                if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    issues.append(f'line {line_no}: box extends outside image (inspect truncation)')
                color = tuple(int(v*255) for v in colorsys.hsv_to_rgb((identity*0.618034)%1, 0.8, 1))
                draw.rectangle(box, outline=color, width=3)
                draw.text((max(0, box[0]), max(25, box[1])), f'GT ID {identity:g}', fill=color)
                count += 1
            title = f'{cam} | frame {stem} | GT boxes: {count} | issues: {len(issues)}'
            draw.rectangle((0, 0, width, 24), fill='black')
            draw.text((5, 5), title, fill='white')
            im.save(args.output_dir / f'{cam}_{stem}.jpg')
            panel = im.copy()
            panel.thumbnail((640, 400))
            panels.append(panel)
            report.append(dict(camera=cam, frame=stem, image=str(image_path),
                               label=str(label_path), boxes=count, issues=issues))
            print(title)
            for issue in issues:
                print('  ' + issue)
        cols = min(3, len(panels))
        grid = Image.new('RGB', (cols*640, math.ceil(len(panels)/cols)*400))
        for i, panel in enumerate(panels):
            grid.paste(panel, ((i % cols)*640, (i // cols)*400))
        grid.save(args.output_dir / f'grid_{stem}.jpg')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Overlays and report saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
