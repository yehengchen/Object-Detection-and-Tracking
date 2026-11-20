"""Regression tests for the assignment helper used by the DeepSORT trackers.

Guards the fix for issue #92: ``linear_assignment`` must keep returning an
Nx2 array of (row, col) index pairs - the format of the removed
``sklearn.utils.linear_assignment_`` - no matter that scipy's
``linear_sum_assignment`` returns a tuple of two arrays.

Run with the tracker project as the working directory:

    cd OneStage/yolo/deep_sort_yolov3
    python ../tests/test_linear_assignment.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.getcwd())


class LinearAssignmentTest(unittest.TestCase):
    def test_wrapper_returns_n_by_2_index_array(self):
        from deep_sort.linear_assignment import linear_assignment

        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=np.float64)
        indices = linear_assignment(cost)

        self.assertEqual(getattr(indices, 'ndim', None), 2,
                         'linear_assignment must not return the scipy tuple')
        self.assertEqual(indices.shape, (3, 2))
        self.assertEqual(sorted(map(tuple, indices)),
                         sorted([(0, 1), (1, 0), (2, 2)]))

    def test_min_cost_matching_end_to_end(self):
        from deep_sort.linear_assignment import min_cost_matching

        cost = np.array([[0.1, 0.9], [0.9, 0.2]], dtype=np.float64)

        def metric(tracks, detections, track_indices, detection_indices):
            return cost

        matches, unmatched_tracks, unmatched_detections = min_cost_matching(
            metric, 0.5, [None, None], [None, None])

        self.assertEqual(sorted(matches), [(0, 0), (1, 1)])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
