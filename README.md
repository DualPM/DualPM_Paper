<div align="center">
  <h1>DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction</h1>
  
  <p align="center">CVPR 2025 - <strong>Conference Highlight</strong></p>
  
  <a href="https://dualpm.github.io">
    <img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-gray.svg">
  </a>
  <a href="https://arxiv.org/abs/2412.04464">
    <img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2412.04464-B31B1B.svg">
  </a>
  
  <br>

  [Ben Kaye*](https://dualpm.github.io), [Tomas Jakab*](https://www.robots.ox.ac.uk/~tomj/), [Shangzhe Wu](https://elliottwu.com), [Christian Rupprecht](https://chrirupp.github.io), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)
  <br>
  <em>(*equal contribution)</em>
  <br>
  [University of Oxford](https://www.robots.ox.ac.uk/~vgg/)
</div>

https://github.com/user-attachments/assets/41ce522e-92b7-463a-9983-48312dadd4be


## 📖 Overview

**DualPM** is a novel approach for 3D shape and pose reconstruction using dual posed-canonical point maps. Dual Point Maps reduce tasks in analysis of deformable objects to mapping an image of an object to its **Dual Point Maps**—a pair of point maps defined in camera space and canonical space. DualPMs are easy to predict with a neural network, enabling effective 3D object reconstruction and other tasks.

## 🚀 Installation

> **Note:** Requires a valid CUDA install to use nvdiffrast.

```bash
git clone https://github.com/ben-kaye/DualPM_Paper.git
cd DualPM_Paper
pip install (-e) .
```

## 📚 Downloads

See `DOWNLOAD.md`.

## 📚 Usage

We provide a separate library, [dualpm_lib](https://github.com/ben-kaye/dualpm_lib), for computing Dual Point Maps, which is automatically installed and is a thin wrapper built upon [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

Example usage is given in `datasets.py > MeshToDualPointmap.calculate_dual_pointmap(.)`.

We train our example model on paired input images and rasterized point map targets.

### Training

1. Download the dataset
2. Configure weight path and data locations in `configs/main.yaml`
3. Run the training script:

```bash
python scripts/train.py
```

### Inference

Inference requires our [feature extraction code](https://github.com/DualPM/dualpm_features) (a thin wrapper of [sd-dino](https://github.com/Junyi42/sd-dino)) to obtain the ODISE-DINO features. This must be computed offline as the models are large.

Complete the feature extraction then use our provided script to obtain predictions:

```bash
python scripts/infer.py
```

### Benchmarking

Our main benchmarking code is available at [AnimodelPoints](https://github.com/DualPM/AnimodelPoints). This is a derivative of the [Animodel benchmark](https://github.com/tomasjakab/animodel) but configured for point clouds.


## 📄 Citing

If you find this work useful, please cite our paper:

```bibtex
@InProceedings{kaye2025dualpm,
  author    = {Ben Kaye and Tomas Jakab and Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title     = {{DualPM}: Dual {Posed-Canonical} Point Maps for {3D} Shape and Pose Reconstruction},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition ({CVPR})},
  month     = {June},
  year      = {2025},
  pages     = {6425--6435}
}
```
