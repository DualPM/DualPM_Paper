# DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction

Main project repository for DualPM, presented at CVPR 2025!

https://dualpm.github.io
```
@InProceedings{Kaye2025,
  author = {Ben Kaye and Tomas Jakab and Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title = {DualPM: Dual Posed‑Canonical Point Maps for 3D Shape and Pose Reconstruction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {6425--6435}
}
```

## Installation

Requires a valid CUDA install to use nvdiffrast.
```
$ git clone https://github.com/ben-kaye/DualPM_Paper.git
$ cd DualPM_Paper
$ pip install (-e) .
```
## Point maps from a scene
We provide a separate library for computing Dual Point Maps[https://github.com/ben-kaye/dualpm_lib], which is automatically installed and is a thin wrapper built apon https://github.com/NVlabs/nvdiffrast.

Example usage is given in `datasets.py >  MeshToDualPointmap.calculate_dual_pointmap(.)`.

We train our example model on paired input images and rasterized point map targets.


## Training

Download the dataset. Configure weight path and data locations in `configs/main.yaml`

Then simply run..
```
python scripts/train.py
```
## Inference

Inference requires our feature extraction code (a thin wrapper of https://github.com/Junyi42/sd-dino) to obtain the ODISE-DINO features. This must be computed offline as the models are large.

https://github.com/ben-kaye/dpm-extractor2/tree/py39

Complete the feature extraction then use our provided script to obtain predictions.
```
python scripts/infer.py
```

## Benchmarking

Our main benchmarking code is available at https://github.com/ben-kaye/dualpm-benchmark
This is a derivative of the Animodel benchmark (https://github.com/tomasjakab/animodel) but configured for point clouds. We refer to it as Animodel Points

PASCAL benchmarking is available in `scripts/pascal.py`.
