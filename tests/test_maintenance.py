"""CPU-only regressions; no TensorFlow, GPU, model weights or video GUI needed.

Run with: python -m unittest discover -s tests -v
Requires NumPy; SciPy integration tests are skipped when SciPy is unavailable.
Assignment boundary tests use a tiny exhaustive solver returning SciPy's tuple
contract, not a replacement shipped with the application.
Video tests execute the real main function with
external devices/model calls mocked; they are not inference integration tests.
"""
import ast
import importlib
import importlib.util
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, mock_open

import numpy as np


YOLO = Path(__file__).resolve().parents[1] / 'OneStage' / 'yolo'


def load_matching(version):
    name = '_maintenance_' + version
    directory = YOLO / ('deep_sort_' + version) / 'deep_sort'
    spec = importlib.util.spec_from_file_location(
        name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + '.linear_assignment')


class AssignmentBoundaryTests(unittest.TestCase):
    def load_module(self, version):
        path = YOLO / ('deep_sort_' + version) / 'deep_sort' / 'linear_assignment.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in parsed.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'min_cost_matching')

        def tuple_solver(costs):
            # Independent, exhaustive fixture for these <=3x3 matrices only.
            # Return the real public SciPy API shape: (row_ids, column_ids).
            nrows, ncols = costs.shape
            if nrows <= ncols:
                candidates = [list(zip(range(nrows), cols))
                              for cols in itertools.permutations(range(ncols), nrows)]
            else:
                candidates = [list(zip(rows, range(ncols)))
                              for rows in itertools.permutations(range(nrows), ncols)]
            pairs = min(candidates, key=lambda pairs: sum(costs[r, c] for r, c in pairs))
            return np.array([r for r, c in pairs]), np.array([c for r, c in pairs])

        namespace = {'np': np, 'linear_sum_assignment': tuple_solver}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        return SimpleNamespace(min_cost_matching=namespace['min_cost_matching'])

    def check_case(self, matrix, expected, track_indices=None, detection_indices=None):
        for version in ('yolov3', 'yolov4'):
            with self.subTest(version=version, matrix=matrix):
                module = self.load_module(version)
                costs = np.array(matrix, dtype=float)
                tracks = [object()] * costs.shape[0]
                detections = [object()] * costs.shape[1]
                actual = module.min_cost_matching(
                    lambda *args: costs.copy(), 0.5, tracks, detections,
                    track_indices=track_indices, detection_indices=detection_indices)
                self.assertEqual([sorted(x) for x in actual], [sorted(x) for x in expected])

    def test_square(self):
        self.check_case([[0.1, 0.4], [0.3, 0.1]], ([(0, 0), (1, 1)], [], []))

    def test_more_detections(self):
        self.check_case([[0.1, 0.3, 0.4], [0.3, 0.1, 0.4]],
                        ([(0, 0), (1, 1)], [], [2]))

    def test_more_tracks(self):
        self.check_case([[0.1, 0.3], [0.3, 0.1], [0.4, 0.4]],
                        ([(0, 0), (1, 1)], [2], []))

    def test_threshold_rejects_assignment(self):
        self.check_case([[0.8, 0.9], [0.9, 0.1]], ([(1, 1)], [0], [0]))

    def test_index_mapping(self):
        self.check_case([[0.1, 0.4], [0.3, 0.1]],
                        ([(8, 5), (2, 9)], [], []), [8, 2], [5, 9])

    def test_empty_inputs_skip_solver(self):
        for version in ('yolov3', 'yolov4'):
            for track_indices, detection_indices in (([], [2]), ([4], []), ([], [])):
                with self.subTest(version=version, tracks=track_indices, detections=detection_indices):
                    metric = MagicMock(side_effect=AssertionError('metric must not run'))
                    actual = self.load_module(version).min_cost_matching(
                        metric, 0.5, [], [], track_indices, detection_indices)
                    self.assertEqual(actual, ([], track_indices, detection_indices))


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy not installed: integration not verified')
class SciPyIntegrationTests(AssignmentBoundaryTests):
    def load_module(self, version):
        return load_matching(version)


class VideoTests(unittest.TestCase):
    def execute_main(self, write_video):
        path = YOLO / 'deep_sort_yolov4' / 'main.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        # Load the real function without importing legacy GPU libraries or
        # opening an InteractiveSession in this CPU-only regression suite.
        function = next(node for node in parsed.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'main')
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
        capture.get.return_value = 2
        cv2 = MagicMock()
        cv2.VideoCapture.return_value = capture
        cv2.waitKey.return_value = 0
        track = SimpleNamespace(track_id=1, time_since_update=0,
                                is_confirmed=lambda: True, to_tlbr=lambda: [0, 0, 1, 1])
        tracker = MagicMock()
        tracker.tracks = [track]
        yolo = MagicMock()
        yolo.detect_image.return_value = ([], [], ['person'])
        clock = MagicMock()
        clock.time.side_effect = iter(range(20))
        opened = mock_open()
        namespace = {
            'np': np, 'cv2': cv2, 'time': clock, 'args': {'input': 'test.avi'},
            'gdet': SimpleNamespace(create_box_encoder=lambda *a, **k: lambda *a: []),
            'nn_matching': MagicMock(), 'Tracker': lambda *a: tracker,
            'Image': SimpleNamespace(fromarray=lambda x: x),
            'Detection': MagicMock(),
            'preprocessing': SimpleNamespace(non_max_suppression=lambda *a: []),
            'pts': {1: []}, 'COLORS': np.zeros((200, 3), dtype=np.uint8),
            'open': opened, 'print': MagicMock(),
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(path), 'exec'), namespace)
        namespace['main'](yolo, write_video=write_video)
        capture.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()
        return cv2, opened

    def test_each_frame_written_once(self):
        cv2, opened = self.execute_main(True)
        self.assertEqual(cv2.VideoWriter.return_value.write.call_count, 2)
        cv2.VideoWriter.return_value.release.assert_called_once()
        opened.return_value.close.assert_called_once()
        calls = [call.args[0] for call in opened.return_value.write.call_args_list]
        self.assertEqual(calls[::4], ['-1,', '0,'])

    def test_disabled_video_does_not_access_output_handles(self):
        cv2, opened = self.execute_main(False)
        cv2.VideoWriter.assert_not_called()
        opened.assert_not_called()


class ConverterTests(unittest.TestCase):
    def test_converter_uses_existing_anchor_filename_and_model_directory(self):
        path = YOLO / 'deep_sort_yolov4' / 'convert.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        main = next(node for node in parsed.body if isinstance(node, ast.If))
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in main.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name)
                  and isinstance(node.value, ast.Constant)}
        self.assertEqual(values['anchors_path'], 'model_data/yolo_anchors.txt')
        self.assertEqual(values['weights_path'], 'model_data/yolov4.weights')


if __name__ == '__main__':
    unittest.main()
