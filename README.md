# On-the-go Reflectance Transformation Imaging with Ordinary Smartphones

A Python toolkit for performing Reflectance Transformation Imaging (RTI) using ordinary smartphones, based on the paper by Mara Pistellato and Filippo Bergamasco[^1]. RTI enables interactive relighting and surface analysis of objects by capturing their appearance under varying lighting conditions.

## Features

- **Camera Calibration**: Calibrate your smartphone camera for accurate RTI.
- **Video Analysis**: Extract lighting and surface information from video sequences.
- **RTI Function Fitting**: Fit mathematical models (RBF, POLY) to captured data.
- **Manual Relighting**: Interactively relight objects using fitted RTI data.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/mr0bz/rti-on-the-go.git
    cd rti-on-the-go
    ```

2. **Install dependencies:**
    - It is recommended to use a virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Install required packages using [mamba](https://mamba.readthedocs.io/en/latest/):
        ```bash
        mamba install numpy opencv scipy matplotlib librosa
        ```
      If you don't have `mamba`, you can install it with:
        ```bash
        conda install mamba -c conda-forge
        ```
    - Alternatively, you can use `pip`:
        ```bash
        pip install numpy opencv-python scipy matplotlib librosa
        ```

## Usage

### 1. Camera Calibration

Calibrate your camera using a calibration video:

```bash
python rti_calibratecamera.py <calibration_video> \
    [-s {linspace,random}] \
    [-n NUM_FRAMES] \
    [-c WIDTH HEIGHT] \
    [-o MTX_OUTPUT_PATH] \
    [-d DIST_OUTPUT_PATH]
```

### 2. Video Analysis

Analyze static and moving light videos to extract RTI data:

```bash
python rti_analyse.py <static_path> <moving_path> <calibration_path> <distortion_path> \
    [-m MARKER_DIMENSION] \
    [-o OUTPUT_PATH] \
    [-d | --debug | --no-debug]
```

### 3. Fit RTI Function

Fit an RTI model (RBF or POLY) to the extracted data:

```bash
python rti_fit.py <input_data.npz> [-i {RBF,POLY}] [-g GRID_SIZE]
```

### 4. Manual Relighting

Interactively relight the object using the fitted RTI model:

```bash
python rti_relighting.py <fitted_model.npz>
```

## Example Workflow

Example using coin #1 (adjust filenames for other objects):

```bash
# 1. Calibrate camera
python rti_calibratecamera.py "./data/cam2 - moving light/calibration.mp4"

# 2. Analyse videos
python rti_analyse.py \
    "./data/cam1 - static/coin1.mov" \
    "./data/cam2 - moving light/coin1.mp4" \
    "./output/K.npy" \
    "./output/dist.npy" \
    --debug

# 3. Fit function (e.g., RBF)
python rti_fit.py "./output/coin1.npz" -i RBF

# 4. Manual relighting
python rti_relighting.py "output/F_coin1_RBF.npz"
```

## References

[^1]: [On-the-go Reflectance Transformation Imaging with Ordinary Smartphones](https://arxiv.org/abs/2210.09821)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Based on the research by Mara Pistellato and Filippo Bergamasco.