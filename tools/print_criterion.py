#!/usr/bin/env python3
"""Standalone checkpoint criterion inspector.

Builds a model, loads checkpoint weights, then prints uncertainty-related
criterion weights for a selected camera criterion.
"""

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from util.tool import load_model  # noqa: E402


def _infer_flags_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', {})

    inferred = {
        'use_uncertainty_loss': False,
        'use_reid_query': False,
        'num_cams': None,
    }

    inferred['use_uncertainty_loss'] = any(
        key.endswith('log_var_tracking') or key.endswith('log_var_reid')
        for key in state.keys()
    )
    inferred['use_reid_query'] = any('reid_module.' in key for key in state.keys())

    cam_ids = []
    for key in state.keys():
        match = re.match(r'criterion\.criteria\.(\d+)\.', key)
        if match is not None:
            cam_ids.append(int(match.group(1)))
    if cam_ids:
        inferred['num_cams'] = max(cam_ids) + 1

    return inferred


def _to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        'Print criterion weights from checkpoint',
        parents=[get_args_parser()],
    )
    parser.set_defaults(meta_arch='multiview_motr', dataset_file='e2e_mv_mot', device='cpu')
    parser.add_argument('--cam_idx', type=int, default=0,
                        help='Camera criterion index to inspect (default: 0)')
    args = parser.parse_args()

    if not args.resume:
        raise ValueError('Please provide --resume <checkpoint_path>.')

    inferred = _infer_flags_from_checkpoint(args.resume)
    if inferred['use_uncertainty_loss'] and not getattr(args, 'use_uncertainty_loss', False):
        args.use_uncertainty_loss = True
    if inferred['use_reid_query'] and not getattr(args, 'use_reid_query', False):
        args.use_reid_query = True
    if inferred['num_cams'] is not None and getattr(args, 'num_cams', 1) == 1:
        args.num_cams = inferred['num_cams']

    model, criterion, _ = build_model(args)
    model = load_model(model, args.resume)

    model_criterion = getattr(model, 'criterion', criterion)
    criteria = getattr(model_criterion, 'criteria', None)

    if criteria is not None and len(criteria) > 0:
        cam_idx = max(0, min(int(args.cam_idx), len(criteria) - 1))
        cam_criterion = criteria[cam_idx]
        path_prefix = f'model.criterion.criteria[{cam_idx}]'
    else:
        cam_idx = 0
        cam_criterion = model_criterion
        path_prefix = 'model.criterion'

    w1 = getattr(cam_criterion, 'w1', None)
    w2 = getattr(cam_criterion, 'w2', None)

    if w1 is None:
        w1 = getattr(cam_criterion, 'log_var_tracking', None)
    if w2 is None:
        w2 = getattr(cam_criterion, 'log_var_reid', None)

    print('--- Criterion Debug ---')
    print(f'meta_arch={args.meta_arch}, dataset_file={args.dataset_file}, num_cams={getattr(args, "num_cams", None)}')
    print(f'inferred: use_uncertainty_loss={inferred["use_uncertainty_loss"]}, '
          f'use_reid_query={inferred["use_reid_query"]}, '
          f'num_cams={inferred["num_cams"]}')
    print(f'{path_prefix}.w1 = {_to_float(w1)}')
    print(f'{path_prefix}.w2 = {_to_float(w2)}')


if __name__ == '__main__':
    main()
