---
license: apache-2.0
base_model: Fengxue93/DepthART
base_model_relation: quantized
pipeline_tag: depth-estimation
tags:
  - depth-estimation
  - monocular-depth
  - webgpu
  - typegpu
---

# DepthART relative-depth, converted for TypeGPU

DepthART relative-depth 448 checkpoints in the `.depthart` container read by
[TypeGPU](https://github.com/software-mansion/TypeGPU).

Unofficial conversion, not affiliated with or endorsed by the DepthART authors.
Original weights: [Fengxue93/DepthART](https://huggingface.co/Fengxue93/DepthART).

## Files

| file | bytes | SHA-256 |
| --- | --- | --- |
| `depthart-relative-s-448-balanced.depthart` | 13,662,992 | `e6d7b65bd2888771790d3cc3ad827133f0b014f05010347b6fc6fc891ff9e19c` |
| `depthart-relative-s-448-f32.depthart` | 23,994,512 | `adc5352f2fc83d1fd7e740ed32b8a0bd7862cef463a430d23d6071990e822aef` |
| `depthart-relative-b-448-balanced.depthart` | 25,518,768 | `cf121c7df9ae5fa5b24a8ae910af8462f1be9bde8131a9e4e5604f902f12b46d` |
| `depthart-relative-b-448-f32.depthart` | 45,445,776 | `a3d16e35ac91f753e7027bee7f4ae13b0007183df1a141947ce80b2d55a45a30` |
| `depthart-relative-l-448-balanced.depthart` | 71,566,624 | `2d39ab90a76039586c1475ec11a467cd789e455e320c56f2836a2390b28be33b` |

`balanced` requires the WebGPU `shader-f16` feature. `f32` requires no optional
features.

## What changed

Not a repackaging. Batch normalization and reparameterizable branches are
folded, graph-only view operations lowered, compatible activations and
channel-affine operations fused.

`balanced` stores and computes selected encoder and decoder tensors in FP16.
Selective-scan recurrence, normalization, residual endpoints, public input and
output, and quality-sensitive reconstruction stay FP32. Output differs
numerically from the original checkpoint.

## Output

Nonnegative affine-invariant relative disparity at 448x448, not metric depth.

Upstream `relative_b_448` emits inverted disparity, near surfaces low and far
surfaces high, opposite of `s` and `l`. Each bundle records its polarity in its
manifest.

## Reproducing

The converter, its commands, and the checksums of the accepted upstream ONNX
artifacts are in `tools/depthart` in the TypeGPU repository. Conversion rejects
any artifact whose identity or graph structure does not match the pinned
figures.

## License

Apache 2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

Upstream carries no LICENSE file. Apache-2.0 is declared by the upstream model
repository metadata and is the license these artifacts are redistributed under.
Redistribution must preserve `LICENSE` and `NOTICE`.
