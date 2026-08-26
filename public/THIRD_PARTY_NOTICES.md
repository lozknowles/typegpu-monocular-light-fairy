# Third-party notices

This isolated preview derives its inference and relighting implementation from the official
TypeGPU Monocular Light Injection example at commit
`40dc5c915d9e869daf7c37e4c0c62904f442002d`. TypeGPU is distributed under the MIT License;
the complete upstream license is included as `typegpu-LICENSE.txt`.

The converted DepthART weights are pinned to Hugging Face revision
`913a7c13ddfbd48549279555d1db98172e8e5e0d`. That repository declares Apache License 2.0 and
supplies the complete `LICENSE`, `NOTICE`, and model card under `model-licence/`.

There is a material licensing ambiguity to retain in any future use: the converted-weight
repository declares Apache-2.0, while its NOTICE reports that the upstream DepthART project
repository has no LICENSE file or copyright notice and instead relies on Apache-2.0 metadata at
the upstream model host. This qualification records that ambiguity; it does not resolve it.

The model emits affine-invariant relative disparity. It is not a metric-depth model and does not
measure physical distance.
