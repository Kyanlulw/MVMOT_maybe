import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from demo_multiview import collect_frame_files
from models.multiview.association import mutual_topk_edges, neighbor_support
from models.multiview.multiview import MultiviewMOTR
from models.structures import Instances


def tracker():
    model = MultiviewMOTR.__new__(MultiviewMOTR)
    torch.nn.Module.__init__(model)
    model.track_bases = [SimpleNamespace(filter_score_thresh=.5) for _ in range(3)]
    model.cross_view_reid_match_thresh = .8
    model.cross_view_top_k = 10
    model.cross_view_spatial_neighbors = 5
    model.cross_view_neighbor_thresh = .5
    model._cross_view_local_to_global = {}
    model._cross_view_global_tracks = {}
    model._cross_view_global_proto = {}
    model._next_cross_view_global_id = 0
    model._tmp_track_state = {}
    model.tmp_window_tau1 = 30
    model.reid_queue_bank = None
    return model


def tracks(ids, features):
    result = Instances((100, 100))
    result.obj_idxes = torch.tensor(ids, dtype=torch.long)
    result.scores = torch.ones(len(ids))
    result.reid_embedding = F.normalize(torch.tensor(features, dtype=torch.float32).reshape(len(ids), 2), dim=1)
    return result


class FusionTrackInference(unittest.TestCase):
    def test_postprocess_preserves_reid_through_query_handoff(self):
        model = tracker()
        model.eval()
        model.memory_bank = None
        model.reid_enabled = True
        model.reid_queue_bank = object()
        model._queue_frame_idx = [0]
        model.track_bases = [SimpleNamespace(update=lambda instances: None)]
        model._update_track_query_queue = lambda *args: None

        def empty():
            result = tracks([-1], [[0., 0.]])
            result.remove('reid_embedding')
            result.pred_logits = torch.zeros(1, 1)
            result.pred_boxes = torch.zeros(1, 4)
            result.output_embedding = torch.zeros(1, 2)
            return result

        class Reid(torch.nn.Module):
            def forward(self, track_instances, **kwargs):
                track_instances.reid_embedding = torch.tensor([[0., 1.]])
                return track_instances, None

        class Handoff(torch.nn.Module):
            def forward(self, data):
                return Instances.cat([data['init_track_instances'], data['track_instances']])

        model._generate_empty_tracks = empty
        model.reid_module = Reid()
        model.track_embed = Handoff()
        current = empty()
        current.obj_idxes[:] = 0
        result = model._post_process_single_image(
            {'pred_logits': torch.ones(1, 1, 1), 'pred_boxes': torch.zeros(1, 1, 4),
             'hs': torch.tensor([[[1., 0.]]])}, current, False, 0,
        )['track_instances']
        self.assertTrue(result.has('reid_embedding'))
        torch.testing.assert_close(result.reid_embedding[1], torch.tensor([0., 1.]))
        model.track_bases = [SimpleNamespace(filter_score_thresh=.5)]
        model._associate_cross_view_reid({0: result})
        torch.testing.assert_close(model._cross_view_global_proto[0], torch.tensor([0., 1.]))

    def test_topk_is_global_and_mutual(self):
        features = F.normalize(torch.tensor([[1., 0.], [1., .1], [1., .3]]), dim=1)
        edges = mutual_topk_edges(features, [0, 1, 2], 1, .8)
        self.assertEqual(set(edges), {(0, 1)})
        self.assertEqual(mutual_topk_edges(features, [0, 0, 0], 10, .8), {})

    def test_neighbor_support_requires_distinct_matches(self):
        edges = {(0, 3): 1., (1, 3): 1., (2, 3): 1.}
        self.assertAlmostEqual(neighbor_support([0, 1, 2], [3, 4], edges), 1 / 3)
        edges[(0, 4)] = 1.
        self.assertAlmostEqual(neighbor_support([0, 1, 2], [3, 4], edges), 2 / 3)

    def test_newcomer_cannot_steal_continuing_identity(self):
        model = tracker()
        model._associate_cross_view_reid({0: tracks([9], [[1., 0.]])})
        result = model._associate_cross_view_reid({0: tracks([10, 9], [[1., 0.], [1., 0.]])})
        self.assertEqual(result[0].tolist(), [1, 0])
        for _ in range(20):
            model._associate_cross_view_reid({0: tracks([10, 9], [[1., 0.], [1., 0.]])})
        self.assertEqual(len(model._cross_view_global_tracks[0]), 1)

    def test_view_exclusivity_and_temporal_continuity(self):
        model = tracker()
        result = model._associate_cross_view_reid({0: tracks([0, 1], [[1., 0.], [1., 0.]]),
                                                 1: tracks([0], [[1., 0.]])})
        self.assertNotEqual(*result[0].tolist())
        self.assertEqual(result[0][0], result[1][0])
        later = model._associate_cross_view_reid({0: tracks([0], [[1., 0.]]),
                                                1: tracks([0], [[0., 1.]])})
        self.assertEqual(later[0][0], later[1][0])

    def test_tmp_reactivation_and_expiry(self):
        model = tracker()
        current = {0: tracks([0], [[1., 0.]])}
        ids = model._associate_cross_view_reid(current)
        model._update_tmp_track_state(current, ids, {0: 0})
        empty = {0: tracks([], [])}
        model._update_tmp_track_state(empty, model._associate_cross_view_reid(empty), {0: 1})
        self.assertEqual(model._tmp_track_state[(0, 0)]['active'], 0)
        model._prune_inference_tmp({0: 30})
        self.assertIn((0, 0), model._tmp_track_state)
        current = {0: tracks([1], [[1., 0.]])}
        ids = model._associate_cross_view_reid(current)
        self.assertEqual(ids[0].item(), 0)
        model._update_tmp_track_state(current, ids, {0: 30})
        self.assertEqual(model._tmp_track_state[(0, 0)]['active'], 1)
        model._prune_inference_tmp({0: 61})
        self.assertFalse(model._cross_view_global_proto)
        self.assertFalse(model._cross_view_local_to_global)

    def test_frames_match_by_identifier_and_sort_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            for cam in ['a', 'b']:
                folder = Path(directory) / cam / 'images'
                folder.mkdir(parents=True)
                for frame in ['1.jpg', '10.jpg', '2.jpg']:
                    (folder / frame).touch()
            files, count = collect_frame_files(directory, ['a', 'b'])
            self.assertEqual(count, 3)
            self.assertEqual(files['a'], ['1.jpg', '2.jpg', '10.jpg'])
            (Path(directory) / 'b' / 'images' / '2.jpg').rename(Path(directory) / 'b' / 'images' / '3.jpg')
            with self.assertRaisesRegex(ValueError, 'identifiers differ'):
                collect_frame_files(directory, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
