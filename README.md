# DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction
In proceedings IEEE/CVF CVPR 2025!
https://dualpm.github.io

## Installation
```
$ git clone https://github.com/ben-kaye/DualPM_Paper.git
$ cd DualPM_Paper
$ pip install (-e) .
$ pip install -r requirements.txt
```
## Extracting pointmaps from a 3D scene
A minimal example for generating pointmap training data is provided in ```scripts/render_example.py```. All that is needed is the model and camera parameters corresponding to the image! ([Nvdiffrast](https://github.com/NVlabs/nvdiffrast) does all the lifting). 

Our procedure expands on this by painting both the canonical template vertices' positions (unposed) and the posed vertices' positions corresponding to the input image. This is our Dual Pointmap target!
We render the input image in Blender and export skeleton, instance mask, and camera parameters.
We then extract our pretrained features (DINOv2 + SD2.0) projected into 64 dimensions.

Finally we train the model with features in, DualPM out.
