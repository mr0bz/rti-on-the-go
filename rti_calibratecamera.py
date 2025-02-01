# REFERENCES
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import argparse
import random
import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from pathlib import Path

DEFAULT_CHESSBOARD_SIZE = (6, 9)
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output"


def calibrate_camera(
    video_path: str | Path,
    sampling_method: str = "linspace",
    num_frames=50,
    chessboard: tuple[int, int] = DEFAULT_CHESSBOARD_SIZE,
    mtx_output_path: str | Path = DEFAULT_OUTPUT_PATH / "K.npy",
    dist_output_path: str | Path = DEFAULT_OUTPUT_PATH / "dist.npy",
):
    """Calibrates the camera from a video file by sampling `num_frames` frames.
    Available sampling methods: `linspace` (default), `random`."""

    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    match sampling_method:
        case "random":
            sample_frames_numbers = random.sample(range(total_frames), num_frames | 20)
        case "linspace":
            sample_frames_numbers = np.linspace(
                start=0, stop=total_frames - 1, num=num_frames | 50, dtype=int
            )
        case _:
            raise ("Unsopported sampling method")

    sampled_frames = []
    for n in sample_frames_numbers:
        cap.set(cv.CAP_PROP_POS_FRAMES, n)
        success, frame = cap.read()
        if success:
            sampled_frames.append(frame)

    ret, mtx, dist = calibrate_camera_from_frames(images=sampled_frames, chessboard=chessboard)

    # Ensure output paths exists
    mtx_output_path.parent.mkdir(parents=True, exist_ok=True)
    dist_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(mtx_output_path, mtx)
    np.save(dist_output_path, dist)

    return ret, mtx, dist


def calibrate_camera_from_frames(
    images: list[MatLike], chessboard: tuple[int, int] = DEFAULT_CHESSBOARD_SIZE
):
    """Calibrate the camera given a set of pictures of a chessboard"""

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard[0], 0 : chessboard[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if len(images) == 0:
        return False, None, None

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, *_ = cv.calibrateCamera(objpoints, imgpoints, images[0].shape[:2], None, None)
    return ret, mtx, dist


def main():
    parser = argparse.ArgumentParser(description="Calibrate video.")
    parser.add_argument("filename", type=str, help="Path to the calibration video file.")
    parser.add_argument(
        "-s",
        "--sampling-method",
        type=str,
        help="Sampling method for frame selection.",
        default="linspace",
        choices=["linspace", "random"],
    )
    parser.add_argument("-n", "--num-frames", type=str, help="Number of frames to use for calibration.", default=50)
    parser.add_argument(
        "-c", "--chessboard",
        nargs=2,
        type=int,
        help="Size of calibration chessboard (W, H).",
        metavar=('WIDTH', 'HEIGHT'),
        default=DEFAULT_CHESSBOARD_SIZE,
    )
    parser.add_argument(
        "-o", "--mtx-output-path",
        type=str,
        help="Output path for calibration matrix.",
        default=DEFAULT_OUTPUT_PATH / "Z.npy",
    )
    parser.add_argument(
        "-d", "--dist-output-path",
        type=str,
        help="Output path for distortion matrix.",
        default=DEFAULT_OUTPUT_PATH / "dist.npy",
    )
    
    args = parser.parse_args()

    _, mtx, dist = calibrate_camera(
        args.filename,
        args.sampling_method,
        args.num_frames,
        args.chessboard,
        args.mtx_output_path,
        args.dist_output_path,
    )
    print("Z", mtx)
    print("dist", dist)


if __name__ == "__main__":
    main()
