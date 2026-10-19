# Targeted maintenance regressions

This patch does not migrate TensorFlow/Keras, change GPU requirements, or
claim to resolve all dependency security advisories.

## Changes

- Use the public SciPy assignment API in both Deep SORT variants, converting
  its `(rows, columns)` tuple into the Nx2 array expected by the caller.
- Write each YOLOv4 output frame once. `main(yolo, write_video=False)` no longer
  accesses uninitialized video/log output handles; the default remains enabled.
- Keep the existing `model_data/yolo_anchors.txt` name and load Darknet weights
  from `model_data/yolov4.weights`. Weights must be provided separately.

## CPU-only regression checks

With NumPy installed:

```sh
python -m unittest discover -s tests -v
```

Six assignment-boundary cases execute the actual matching function with an
independent tiny-matrix solver that returns SciPy's public tuple contract.
Two video tests execute the actual main function with external devices and
inference mocked, and one test checks converter path constants.
These tests use Python 3's standard-library AST and unittest APIs; they do
not establish that the legacy application runs on modern Python.

When SciPy is installed, six additional integration cases import the real
Deep SORT module and exercise the real solver. They are explicitly skipped
otherwise. No TensorFlow/GPU inference or ROS integration is covered.

Validation on 2026-10-19: Python 3.12, NumPy 2.3.5; 9 passed, 6 skipped.
SciPy's official wheel download failed HTTPS certificate validation in the
maintenance environment. Certificate verification was not disabled.
Legacy production requirements are unchanged and remain a separate migration
and security-maintenance task.
