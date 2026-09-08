# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------



import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from torch.utils.data import DataLoader
import datasets

from util.motdet_eval import motdet_evaluate, detmotdet_evaluate
from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_mot, train_one_epoch_multiview_mot
from models import build_model


MOT_LIKE_DATASETS = {
    'e2e_mot',
    'e2e_dance',
    'mot',
    'ori_mot',
    'e2e_static_mot',
    'e2e_joint',
    'e2e_mv_mot',
}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _optimizer_param_groups_for_manifest(optimizer):
    param_groups = []
    for group in optimizer.param_groups:
        serializable_group = {
            key: _json_safe(value)
            for key, value in group.items()
            if key != 'params'
        }
        serializable_group['n_parameters'] = sum(
            p.numel() for p in group.get('params', [])
        )
        param_groups.append(serializable_group)
    return param_groups


def _build_training_manifest(args, n_parameters, optimizer, lr_scheduler,
                             train_loader_len, val_loader_len):
    started_at = datetime.datetime.now(datetime.timezone.utc)
    return {
        'schema_version': 1,
        'run_id': started_at.strftime('%Y%m%dT%H%M%SZ'),
        'created_at_utc': started_at.isoformat(),
        'git': utils.get_sha(),
        'command': {
            'python': sys.executable,
            'argv': sys.argv,
        },
        'runtime': {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'pytorch_alloc_conf': os.environ.get('PYTORCH_ALLOC_CONF'),
            'pytorch_cuda_alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            'distributed': getattr(args, 'distributed', False),
            'rank': utils.get_rank(),
            'world_size': utils.get_world_size(),
            'device': args.device,
            'seed': args.seed,
            'rank_seed': args.seed + utils.get_rank(),
        },
        'hyperparameters': _json_safe(vars(args)),
        'model': {
            'meta_arch': args.meta_arch,
            'backbone': args.backbone,
            'num_parameters_trainable': n_parameters,
            'num_queries': args.num_queries,
            'num_cams': getattr(args, 'num_cams', None),
            'use_reid_query': getattr(args, 'use_reid_query', False),
            'use_uncertainty_loss': getattr(args, 'use_uncertainty_loss', False),
        },
        'data': {
            'dataset_file': args.dataset_file,
            'mot_path': getattr(args, 'mot_path', None),
            'data_txt_path_train': getattr(args, 'data_txt_path_train', None),
            'data_txt_path_val': getattr(args, 'data_txt_path_val', None),
            'train_batches_per_epoch': train_loader_len,
            'val_batches': val_loader_len,
        },
        'optimizer': {
            'type': optimizer.__class__.__name__,
            'param_groups': _optimizer_param_groups_for_manifest(optimizer),
        },
        'lr_scheduler': {
            'type': lr_scheduler.__class__.__name__,
            'state': _json_safe(lr_scheduler.state_dict()),
            'interval': args.lr_scheduler_interval,
            'cosine_start_epoch': args.cosine_start_epoch,
        },
    }


def _write_training_manifest(args, n_parameters, optimizer, lr_scheduler,
                             data_loader_train, data_loader_val):
    if not args.output_dir or not utils.is_main_process():
        return None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_training_manifest(
        args, n_parameters, optimizer, lr_scheduler,
        len(data_loader_train), len(data_loader_val),
    )
    run_id = manifest['run_id']
    manifest_path = output_dir / f'training_manifest_{run_id}.json'
    latest_path = output_dir / 'training_manifest_latest.json'

    for path in (manifest_path, latest_path):
        with path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')

    print(f'Wrote training manifest to {manifest_path}')
    return manifest_path


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_scheduler', default='step', type=str, choices=['step', 'cosine'])
    parser.add_argument('--lr_scheduler_interval', default='epoch', type=str, choices=['epoch', 'step'],
                        help='Update LR scheduler every epoch or every optimizer step')
    parser.add_argument('--cosine_start_epoch', default=0, type=int,
                        help='For cosine scheduler: keep LR flat until this epoch, then start cosine decay')
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--meta_arch', default='deformable_detr', type=str)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_anchors', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)
    parser.add_argument('--coco_path', default='/data/workspace/detectron2/datasets/coco/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--smoke_train', action='store_true', default=False,
                        help='Use a small image scale for a one-batch smoke run')
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # end-to-end mot settings.
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--input_video', default='figs/demo.mp4', type=str)
    parser.add_argument('--data_txt_path_train',
                        default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--data_txt_path_val',
                        default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--img_path', default='data/valid/JPEGImages/')

    parser.add_argument('--query_interaction_layer', default='QIM', type=str,
                        help="")
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)
    parser.add_argument('--track_query_queue_len', type=int, default=0,
                        help='Length of per-track FIFO queue used for ReID consumption')
    parser.add_argument('--track_query_queue_score_thresh', type=float, default=0.0,
                        help='Min track score to store a snapshot in track query queue')
    parser.add_argument('--track_query_history_len', type=int, default=0,
                        help='Deprecated alias of --track_query_queue_len')

    parser.add_argument('--use_reid_query', action='store_true', default=False,
                        help='Enable trajectory ReID query module and append ReID loss terms')
    parser.add_argument('--reid_loss_coef', type=float, default=1.0,
                        help='Coefficient for frame_i_reid_total in standard loss summation')
    parser.add_argument('--reid_num_ids', type=int, default=2048,
                        help='Classifier size for ReID head')
    parser.add_argument('--reid_tau1', type=int, default=30,
                        help='Queue depth for ReID trajectory memory')
    parser.add_argument('--reid_tau2', type=int, default=10,
                        help='Window length from queue fed into ReID transformer')
    parser.add_argument('--reid_label_smoothing', type=float, default=0.1,
                        help='Label smoothing used in ReID CE loss')
    parser.add_argument('--reid_vit_dim', type=int, default=256,
                        help='Internal feature dim of lightweight ReID transformer')
    parser.add_argument('--reid_num_layers', type=int, default=12,
                        help='Number of transformer layers in lightweight ReID backbone')
    parser.add_argument('--reid_num_heads', type=int, default=6,
                        help='Attention heads in lightweight ReID backbone')
    parser.add_argument('--reid_dropout', type=float, default=0.1,
                        help='Dropout in lightweight ReID backbone')
    parser.add_argument('--reid_warmup_epochs', type=int, default=20,
                        help='Tracking-only warmup before enabling the ReID objective')
    parser.add_argument('--reid_temporal_decay_alpha', type=float, default=1.0,
                        help='Deprecated OUM decay parameter; ignored by the OUM-free FusionTrack path')
    parser.add_argument('--cross_view_reid_match_thresh', type=float, default=0.8,
                        help='Cosine similarity threshold for linking identities across cameras at inference')
    parser.add_argument('--cross_view_top_k', type=int, default=10)
    parser.add_argument('--cross_view_spatial_neighbors', type=int, default=5)
    parser.add_argument('--cross_view_neighbor_thresh', type=float, default=0.5)
    parser.add_argument('--cross_view_reid_momentum', type=float, default=0.9,
                        help='Deprecated compatibility option; latest descriptors are used for re-entry')

    parser.add_argument('--use_checkpoint', action='store_true', default=False)

    # Multi-view settings
    parser.add_argument('--num_cams', default=1, type=int,
                        help='Number of camera views for multi-view tracking')

    # Uncertainty-based multi-task balancing between tracking and ReID losses.
    parser.add_argument('--use_uncertainty_loss', action='store_true', default=False,
                        help='Enable learnable uncertainty weighting between tracking and ReID losses')
    parser.add_argument('--uncertainty_init_tracking', default=-1.85, type=float,
                        help='Initial value for tracking uncertainty log-variance (w1)')
    parser.add_argument('--uncertainty_init_reid', default=-1.05, type=float,
                        help='Initial value for ReID uncertainty log-variance (w2)')
    parser.add_argument('--uncertainty_init', default=None, type=float,
                        help='Deprecated shared init for both uncertainty log-variances (overrides both when set)')
    # wandb logging
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='MOTR', type=str,
                        help='W&B project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='W&B entity (team or username)')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='W&B run name (defaults to exp_name)')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.dataset_file == 'e2e_mv_mot' and args.meta_arch not in ('multiview_motr', 'fusiontrack_motr'):
        print("Warning: dataset_file=e2e_mv_mot requires a multiview architecture. Overriding meta_arch.")
        args.meta_arch = 'multiview_motr'

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize wandb (only on main process)
    if args.wandb and HAS_WANDB and utils.is_main_process():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.exp_name,
            config=vars(args),
            resume='allow',
        )
    elif args.wandb and not HAS_WANDB:
        print('Warning: wandb is not installed. Run `pip install wandb` to enable logging.')
        args.wandb = False

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    if args.dataset_file in MOT_LIKE_DATASETS:
        collate_fn = utils.mot_collate_fn
    else:
        collate_fn = utils.collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    scheduler_step_per_iter = args.lr_scheduler_interval == 'step'
    cosine_start_epoch = max(0, args.cosine_start_epoch)
    if args.lr_scheduler == 'cosine':
        if scheduler_step_per_iter:
            cosine_t_max = max(1, (args.epochs - cosine_start_epoch) * len(data_loader_train))
        else:
            cosine_t_max = max(1, args.epochs - cosine_start_epoch)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=0.0,
        )
    else:
        step_size = args.lr_drop
        if scheduler_step_per_iter:
            step_size = max(1, args.lr_drop * len(data_loader_train))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'reid_identity_map' in checkpoint and hasattr(model_without_ddp, '_reid_obj_to_cls'):
            model_without_ddp._reid_obj_to_cls = {
                int(k): int(v) for k, v in checkpoint['reid_identity_map'].items()
            }
            model_without_ddp._reid_class_owner = {
                int(v): int(k) for k, v in model_without_ddp._reid_obj_to_cls.items()
            }
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop and args.lr_scheduler == 'step':
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                if scheduler_step_per_iter:
                    lr_scheduler.step_size = max(1, args.lr_drop * len(data_loader_train))
                else:
                    lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    _write_training_manifest(
        args, n_parameters, optimizer, lr_scheduler,
        data_loader_train, data_loader_val,
    )

    print("Start training")
    start_time = time.time()

    train_func = train_one_epoch
    if args.dataset_file in MOT_LIKE_DATASETS:
        train_func = train_one_epoch_mot
        dataset_train.set_epoch(args.start_epoch)
        dataset_val.set_epoch(args.start_epoch)
    if args.dataset_file == 'e2e_mv_mot':
        train_func = train_one_epoch_multiview_mot
        dataset_train.set_epoch(args.start_epoch)
        dataset_val.set_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        # FusionTrack warms up single-view tracking before enabling the
        # cross-view ReID objective, matching the paper's progressive setup.
        model_for_schedule = model.module if hasattr(model, 'module') else model
        if getattr(model_for_schedule, 'use_reid_query', False):
            model_for_schedule.reid_enabled = epoch >= getattr(args, 'reid_warmup_epochs', 20)
            if hasattr(model_for_schedule, 'criterion'):
                model_for_schedule.criterion.reid_enabled = model_for_schedule.reid_enabled
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_func(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
            lr_scheduler=lr_scheduler,
            scheduler_step_per_iter=scheduler_step_per_iter,
            scheduler_start_epoch=cosine_start_epoch if args.lr_scheduler == 'cosine' else 0,
        )
        if not scheduler_step_per_iter:
            if args.lr_scheduler == 'cosine':
                if epoch >= cosine_start_epoch:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

        # Log train stats to wandb
        if args.wandb and HAS_WANDB and utils.is_main_process():
            wandb_log = {f'train/{k}': v for k, v in train_stats.items()}
            wandb_log['epoch'] = epoch
            wandb.log(wandb_log, step=epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_period == 0 or (((args.epochs >= 100 and (epoch + 1) > 100) or args.epochs < 100) and (epoch + 1) % 5 == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'reid_identity_map': {
                        str(k): int(v) for k, v in getattr(
                            model_without_ddp, '_reid_obj_to_cls', {}
                        ).items()
                    },
                }, checkpoint_path)
        
        if args.dataset_file not in MOT_LIKE_DATASETS:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

            # Log test stats to wandb
            if args.wandb and HAS_WANDB and utils.is_main_process():
                wandb_log = {f'val/{k}': v for k, v in test_stats.items()}
                # Log COCO eval AP metrics if available
                if 'coco_eval_bbox' in test_stats:
                    ap_names = ['AP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l',
                                'AR1', 'AR10', 'AR100', 'AR_s', 'AR_m', 'AR_l']
                    for i, name in enumerate(ap_names):
                        if i < len(test_stats['coco_eval_bbox']):
                            wandb_log[f'val/{name}'] = test_stats['coco_eval_bbox'][i]
                wandb.log(wandb_log, step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
        if args.dataset_file in MOT_LIKE_DATASETS:
            dataset_train.step_epoch()
            dataset_val.step_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Finish wandb run
    if args.wandb and HAS_WANDB and utils.is_main_process():
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
