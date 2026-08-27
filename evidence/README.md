# Qualification evidence

This directory intentionally contains diagnostics only, never uploaded photographs or camera
frames. The private tailnet URL is redacted in the committed copies; exact originals remain in the
isolated qualification workspace.

- `qualification-hpubuntu-wing-reflection-20260826.json` records the reproducible static-image
  Intel Gen9 run.
- `qualification-msi-live-wing-reflection-20260826.json` records the MSI Edge live-camera adapter,
  feature, model, timing, and frame-counter results. It explicitly records that no camera screenshot
  or frame was retained.
- `qualification-firefly-flight-20260827.json` records the static hpubuntu and MSI qualifications
  for autonomous banked flight, the single abdomen light, reduced wing caster strength, timed output
  hashes, and no-camera visual checks.
- `qualification-fairy-rotation-20260827.json` records hpubuntu and MSI static-demo evidence for the
  bounded roll-and-pitch projection, pose ranges, time-separated canvas hashes, GPU timings, and the
  explicit no-camera boundary.

The NVIDIA Quadro P5000 result remains blocked because protected workloads occupied GPU memory. No
process was stopped to obtain a benchmark. The full operational evidence remains outside this source
repository in the isolated qualification workspace.
