## Downloading the dataset

This dataset is a derivate of the Animodel https://github.com/tomasjakab/animodel
It's available here: https://www.robots.ox.ac.uk/~vgg/research/dualpm/#section-downloads


Verification: Test that the SHA512 matches the provided `SHA512SUMS`
```
sha512sum * > TEST_SHA512SUMS
```

Navigate to the destination parent folder.
Untar the `dualpm_dataset*.tar` with `tar -xf path/to/dualpm_dataset*.tar`


EG:
```
cd path/to/parent
tar -xf path/to/dualpm_dataset_meta.tar
tar -xf path/to/dualpm_dataset_rgb.tar
tar -xf path/to/dualpm_dataset_feats.tar
```
