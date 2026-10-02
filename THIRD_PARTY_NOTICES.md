# Third-party notices

SiHC-specific code is covered by the root [MIT license](LICENSE). The
third-party components below retain their original attribution and terms.
Upstream copyright notices are retained.

## JiT

The Transformer building blocks, timestep/class conditioning, context-token
organization, initialization, and pixel-space flow conventions in
`sihc/models/sihc/components.py`, `sihc/models/sihc/model.py`, and the training
and sampling utilities build on [JiT](https://github.com/LTH14/JiT).
The SiHC implementation changes the residual carrier, spatial read/write
connections, kernel execution, and attention implementation.

Copyright (c) 2025 Tianhong Li. Distributed under the MIT License; the full
upstream notice is preserved in [LICENSES/JiT.txt](LICENSES/JiT.txt).
[Official license source](https://github.com/LTH14/JiT/blob/main/LICENSE).

## ADM / guided-diffusion

The ImageNet center-crop routine in `sihc/imagenet.py` follows the ADM routine
from [guided-diffusion](https://github.com/openai/guided-diffusion), also used
by JiT. This release updates the Pillow resampling constants and returns
images through its own dataset interfaces.

Copyright (c) 2021 OpenAI. Distributed under the MIT License; the full
upstream notice is preserved in
[LICENSES/guided-diffusion.txt](LICENSES/guided-diffusion.txt).
[Official license source](https://github.com/openai/guided-diffusion/blob/main/LICENSE).

## Installed dependencies

PyTorch, Triton, NumPy, Pillow, torchvision, and optional sihc/evaluation/research
packages are installed separately and keep their own licenses. No third-party
model checkpoints, dataset images, or pretrained weights are included here.

## Research controls and long skip

The controlled JiT/mHC models and archived long-skip pipeline build on the same JiT components described above. Original module names are retained in `research/long_skip/vendor` for checkpoint compatibility. Muon utilities retain their source comments and attribution. External RAE/decoder implementations and pretrained teachers are installed separately rather than vendored.
