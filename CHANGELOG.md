# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow semantic versioning.

## [Unreleased]

### Pending qualification

- Pixel 8 Pro front/rear camera acceptance.
- MSI native file-picker acceptance.
- Human visual judgement of the reflective glasses glint.
- Quadro P5000 FP32 benchmark when sufficient GPU memory is available without disturbing protected
  workloads.

## [0.1.0] - 2026-08-26

### Added

- Standalone local-browser build derived from the official TypeGPU Monocular Light Injection
  example at commit `40dc5c915d9e869daf7c37e4c0c62904f442002d`.
- Exact FP16 and FP32 DepthART model revision, byte-count, SHA-256, licence, NOTICE, and model-card
  pins.
- Demo photograph, local upload, and front/rear camera source paths.
- Camera, relative-disparity, normals, and relit views.
- Secure-context, browser, adapter, WebGPU, `shader-f16`, model, compilation, inference, FPS,
  dropped-frame, and timing-source diagnostics.
- True GPU timestamp-query timing with a clearly labelled wall-clock fallback.
- Local HTTPS static server with restrictive security headers and loopback binding by default.
- Procedural flying fairy with four fluttering wings and three coloured light lobes.
- Analytically projected wing-shaped shadows and a Shadow control.
- Three-lobe reflective-surface glint approximation and a Reflections control.
- Immutable, fail-closed model download and post-build staging scripts.
- Architecture, provenance, privacy, licensing, qualification, and reproducibility documentation.

### Changed

- Camera scheduling now permits only one in-flight GPU operation and counts skipped inputs instead of
  building an unbounded queue.
- Camera benchmarks now measure completed pipeline frames, record cold and warmed inference, and
  reset foreground frame counters correctly.
- Vite was updated to `8.0.16` after a dependency audit identified a vulnerability in the earlier
  pin.

### Qualification

- Passed static WebGPU qualification on hpubuntu Intel Gen9 using FP16 and true GPU timestamps.
- Passed static and front-camera runtime qualification on MSI Edge 151 / Intel Xe-LPG.
- Recorded the Quadro P5000 FP32 attempt as blocked after Vulkan device loss under protected GPU
  memory pressure; no workload was killed or restarted.
- Retained diagnostics only for camera testing; no camera frame or screenshot was committed.
