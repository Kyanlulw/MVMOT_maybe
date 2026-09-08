import json
import unittest
import tempfile
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

from main import get_args_parser, _write_training_manifest
from models.deformable_transformer_plus import DeformableTransformer
from models.multiview.multiview import MultiviewClipMatcher
from util.misc import inverse_sigmoid
from util.tool import load_model
from engine import train_one_epoch_multiview_mot


class FusionTrackRegression(unittest.TestCase):
    def test_training_manifest_written_with_hyperparameters(self):
        with tempfile.TemporaryDirectory() as directory:
            args = get_args_parser().parse_args([])
            args.output_dir = directory
            args.dataset_file = 'e2e_mv_mot'
            args.meta_arch = 'fusiontrack_motr'
            args.device = 'cpu'
            args.distributed = False
            args.rank = 0
            args.world_size = 1
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)

            manifest_path = _write_training_manifest(
                args, 3, optimizer, scheduler, [0, 1], [0],
            )

            latest_path = Path(directory) / 'training_manifest_latest.json'
            self.assertTrue(manifest_path.exists())
            self.assertTrue(latest_path.exists())
            manifest = json.loads(latest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['hyperparameters']['output_dir'], directory)
            self.assertEqual(manifest['model']['meta_arch'], 'fusiontrack_motr')
            self.assertEqual(manifest['data']['train_batches_per_epoch'], 2)
            self.assertEqual(manifest['optimizer']['type'], 'AdamW')
            self.assertEqual(manifest['lr_scheduler']['type'], 'StepLR')

    def test_cosine_starts_after_warmup_and_finishes_at_zero(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, data):
                return {'losses_dict': {'objective': self.weight.square()}}

        model = ToyModel()
        criterion = torch.nn.Module()
        criterion.weight_dict = {'objective': 1.}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        for epoch in range(4):
            train_one_epoch_multiview_mot(
                model, criterion, [{}], optimizer, torch.device('cpu'), epoch,
                lr_scheduler=scheduler, scheduler_step_per_iter=True,
                scheduler_start_epoch=2,
            )
            expected = [.1, .1, .05, 0.][epoch]
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)

    def test_pretrained_detection_weights_preserved_or_bootstrapped(self):
        model = torch.nn.Module()
        model.transformer = torch.nn.Linear(2, 2)
        model.detection_transformer = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            state = {k: torch.full_like(v, 2. if k.startswith('transformer.') else 7.)
                     for k, v in model.state_dict().items()}
            torch.save({'model': state}, path)
            load_model(model, path)
            torch.testing.assert_close(model.detection_transformer.weight,
                                       torch.full((2, 2), 7.))
            torch.save({'model': {k: v for k, v in state.items()
                                 if k.startswith('transformer.')}}, path)
            load_model(model, path)
            torch.testing.assert_close(model.detection_transformer.weight,
                                       model.transformer.weight)

    def test_uncertainty_counts_reid_once(self):
        criterion = MultiviewClipMatcher(
            1, 1, None,
            {'frame_0_loss_ce': 1., 'frame_0_reid_total': 1.,
             'clip_reid_triplet': 1.}, [], True, 0., 0.,
        )
        criterion.criteria[0].num_samples = 1
        criterion.criteria[0].sample_device = torch.device('cpu')
        tracking = torch.tensor(2., requires_grad=True)
        ce = torch.tensor(3., requires_grad=True)
        triplet = torch.tensor(4., requires_grad=True)
        cross = torch.tensor(5., requires_grad=True)
        result = criterion({0: {'losses_dict': {
            'frame_0_loss_ce': tracking, 'frame_0_reid_total': ce + triplet,
            'frame_0_reid_ce': ce, 'frame_0_reid_triplet': triplet,
            'clip_reid_triplet': cross,
        }}})['loss_uncertainty_total']
        self.assertAlmostEqual(result.item(), 7.)
        result.backward()
        for term in (tracking, ce, triplet, cross):
            self.assertAlmostEqual(term.grad.item(), .5)

    def test_shared_memory_reference_roundtrip_and_gradients(self):
        torch.manual_seed(4)
        model = DeformableTransformer(
            d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=64, dropout=0., num_feature_levels=1,
            return_intermediate_dec=True,
        )
        src = [torch.randn(1, 32, 4, 4)]
        masks = [torch.zeros(1, 4, 4, dtype=torch.bool)]
        pos = [torch.randn_like(src[0])]
        queries = torch.randn(3, 64)
        calls = []
        handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        memory = model.encode(src, masks, pos)
        det = model.decode(memory, queries)
        refs = torch.tensor([[.05, .95], [.2, .8], [.4, .6]])
        tracking = model.decode(memory, queries, inverse_sigmoid(refs))
        torch.testing.assert_close(tracking[1][0], refs)
        self.assertEqual(len(calls), 1)
        (det[0].square().sum() + tracking[0].square().sum()).backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        handle.remove()

    def test_checkpoint_dropout_gradient_parity(self):
        torch.manual_seed(1)
        layer = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.3))
        x = torch.randn(2, 8)
        torch.manual_seed(8)
        layer(x).square().sum().backward()
        expected = [p.grad.clone() for p in layer.parameters()]
        layer.zero_grad()
        torch.manual_seed(8)
        checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True).square().sum().backward()
        for p, grad in zip(layer.parameters(), expected):
            torch.testing.assert_close(p.grad, grad)


if __name__ == '__main__':
    unittest.main()
