# On-the-go Reflectance Transformation Imaging with Ordinary Smartphones

> Based on the paper of the same name[^1] from Mara Pistellato and Filippo Bergamasco.

Reflectance Transformation Imaging (RTI) is a technique to capture the reflectance of an object under different lighting conditions. It enables the interactive relighting of the subject from any direction and the mathematical enhancement of its surface shape and color attributes.

[^1]: [On-the-go Reflectance Transformation Imaging with Ordinary Smartphones](https://arxiv.org/abs/2210.09821)


## Examples

### Camera calibration

```bash
$ python rti_calibratecamera.py [-h] filename \
    [-s {linspace,random}] \
    [-n NUM_FRAMES] \
    [-c WIDTH HEIGHT] \
    [-o MTX_OUTPUT_PATH] \
    [-d DIST_OUTPUT_PATH]
```

### Video analysis

```bash
$ python rti_analyse.py [-h] \
    static_path \
    moving_path \
    calibration_path \
    distortion_path \
    [-m MARKER_DIMENSION] \
    [-o OUTPUT_PATH] \
    [-d | --debug | --no-debug]
```

### Fitting RTI function

```bash
$ python rti_fit.py [-h] filename [-i {RBF,POLY}] [-g GRID_SIZE]
```

### Manual relighting

```bash
$ python rti_relighting.py [-h] filename
```


## Full example with exam files

Example with coin #1, rename filenames for the other coins.

### 1. Calibrate camera

```bash
$ python rti_calibratecamera.py "./data/cam2 - moving light/calibration.mp4"
```

### 2. Analyse videos

```bash
$ python rti_analyse.py \
    "./data/cam1 - static/coin1.mov" \
    "./data/cam2 - moving light/coin1.mp4" \
    "./output/K.npy" \
    "./output/dist.npy" \
    --debug
```

### 3. Fit function

Example with RBF.

```bash
$ python rti_fit.py "./output/coin1.npz" -i RBF
```

### 4. Manual relighting

```bash
$ python rti_relighting.py "output/F_coin1_RBF.npz"
```