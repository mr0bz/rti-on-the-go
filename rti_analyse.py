import numpy as np
import cv2 as cv
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from cv2.typing import MatLike

############# DEBUGGING AND TESTING PARAMETERS ############
FLG_DEBUG = False
NUM_VIDEO = 1  # Choose test number (1 to 4)

################### INTERNAL PARAMETERS ###################
SQUARE_DETECTION_BLUR_KERNEL = (5, 5)
SQUARE_DETECTION_BLUR_SIGMA = 1.5
SQUARE_DETECTION_CONTOURS_EPS = 0.005
SQUARE_NOT_FOUND = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
CIRCLE_NOT_FOUND = np.array([-1, -1])
DEFAULT_MARKER = np.array([[0, 0, 1], [500, 0, 1], [500, 500, 1], [0, 500, 1]])
DEFAULT_MARKER_CIRCLE = np.array([-43.29842, 546.0332])

####################### DEBUG UTILS #######################
COLOR_BLUE = [255, 0, 0]
COLOR_CYAN = [255, 255, 0]
COLOR_RED = [0, 0, 255]
COLOR_GREEN = [0, 255, 0]

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output"


def draw_light_direction(u, v):
    img = np.zeros((400, 400), dtype=np.uint8)
    cv.circle(img, (200, 200), 200, (255, 255, 255), 2, cv.LINE_AA)

    x = int(200 + u * 200)
    y = int(200 + v * 200)

    cv.line(img, (200, 200), (x, y), (255, 255, 255), 2, cv.LINE_AA)
    cv.circle(img, (x, y), 20, (255, 255, 255), 2, cv.LINE_AA)

    return img



def draw_debug(img, square, circle):
    if not np.array_equal(square, SQUARE_NOT_FOUND):
        cv.drawContours(img, [square], -1, COLOR_RED, 3)
        for i, point in enumerate(square):
            cv.circle(img, point, 5, COLOR_GREEN, -1)
            cv.putText(
                img,
                f"[{i}]: x={point[0]}, y={point[1]}",
                point + [20, 0],
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                COLOR_CYAN,
                2,
            )

    if not np.array_equal(circle, CIRCLE_NOT_FOUND):
        cv.drawMarker(img, circle, COLOR_GREEN, cv.MARKER_CROSS, thickness=2)

    return img


def detect_square(img: MatLike):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, SQUARE_DETECTION_BLUR_KERNEL, SQUARE_DETECTION_BLUR_SIGMA)

    # highlight black features (the square) in order to attenuate the shadow
    # NB: tophat flips blacks and whites in the image
    se = cv.getStructuringElement(cv.MORPH_RECT, (200, 200))
    bh = cv.morphologyEx(blur, cv.MORPH_BLACKHAT, se)

    _, otsu = cv.threshold(bh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Hierarchical closed contour extraction
    contours, _ = cv.findContours(otsu, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Ramer-Douglas-Peucker algorithm for contour approximation

    approx = [
        cv.approxPolyDP(c, SQUARE_DETECTION_CONTOURS_EPS * cv.arcLength(c, True), True)
        for c in contours
    ]

    # Keep only contours with four corners (possible squares)
    # Sort by area in order to identify the two biggest ones
    squares = [a.squeeze() for a in approx if len(a) == 4]

    internal_square = SQUARE_NOT_FOUND
    circle = CIRCLE_NOT_FOUND
    circle_pos = None
    if len(squares) >= 2:
        [internal_square, external_square] = sorted(squares, key=cv.contourArea)[-2:]

        # STEP 1. Align internal and external squares corners
        # Given the findContour algorithm implemented by opencv, we assume that
        #   the corners of the external square are extracted counter-clockwise,
        #   while the corners of the internal square are extracted clockwise,
        #   beginning with the corner in the top-left.
        # Source: Paper 'Topological Structural Analysis of Digitized Binary Images by Border Following', Appendix 1
        # Useful: https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation

        # STEP 2. Find corners pair near to the white circle
        circles_found = 0
        for idx in range(4):
            # Circle is not exactly midway between the two corners
            # Displacement values computed from the marker.svg, for better debug visualization
            # true_midpoint = np.round((internal_square[idx] + external_square[(4-idx)%4]) / 2).astype(int)
            i = internal_square[idx]
            e = external_square[(4 - idx) % 4]
            midpoint = np.round(i + (e - i) * [0.4329842, 0.460332]).astype(int)

            # Check if midpoint is black (because colors were inverted by blackhat)
            if otsu[*midpoint[::-1]] == 0:
                circle = midpoint
                circle_pos = idx
                circles_found += 1

        # STEP 3. Sort internal square corners accordingly
        if (
            circles_found == 1
            and not np.array_equal(circle, CIRCLE_NOT_FOUND)
            and circle_pos is not None
        ):
            internal_square = np.roll(internal_square, -circle_pos, axis=0)
        else:
            internal_square = SQUARE_NOT_FOUND
            circle = CIRCLE_NOT_FOUND

    return internal_square, circle


def get_videos_sync_time(video1_path: str | Path, video2_path: str | Path) -> tuple[int, int]:
    audio1, sr = librosa.load(Path(video1_path))
    audio2, sr = librosa.load(Path(video2_path))
    correlation = signal.correlate(audio1, audio2, mode="full")
    lags = signal.correlation_lags(audio1.size, audio2.size, mode="full")
    lag = lags[np.argmax(correlation)]

    if lag < 0:
        return 0, -lag / sr

    return lag / sr, 0



def analyse(
    static_video_path: str | Path,
    moving_video_path: str | Path,
    calibration_matrix_npy: str | Path,
    distortion_matrix_npy: str | Path,
    marker=DEFAULT_MARKER,
    debug=False,
):
    # Load calibration matrix
    K = np.load(calibration_matrix_npy)
    dist = np.load(distortion_matrix_npy)

    # Get video from both cams
    static = cv.VideoCapture(static_video_path)
    moving = cv.VideoCapture(moving_video_path)

    # Get FPS for both videos, needed for sync loop
    static_fps = static.get(cv.CAP_PROP_FPS)
    moving_fps = moving.get(cv.CAP_PROP_FPS)

    # TODO: Sync videos by audio --> set starting frames
    static_start, moving_start = get_videos_sync_time(static_video_path, moving_video_path)
    moving_delay = moving_start - static_start
    static.set(cv.CAP_PROP_POS_FRAMES, np.round(static_start * static_fps))
    moving.set(cv.CAP_PROP_POS_FRAMES, np.round(moving_start * moving_fps))

    MLIC = []
    L = []
    U = []
    V = []

    if debug:
        # Niceties for better visualization on 2K screen
        cv.namedWindow("Static camera", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow("Moving camera", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow("Static camera warped", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow("Light direction", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)

        cv.resizeWindow("Static camera", 1080 // 2, 1920 // 2)
        cv.resizeWindow("Moving camera", 1920 // 2, 1080 // 2)
        cv.resizeWindow("Static camera warped", 375, 375)
        cv.resizeWindow("Light direction", 375, 375)

        cv.moveWindow("Static camera", 20, 0)
        cv.moveWindow("Moving camera", 1080 // 2 + 40, 0)
        cv.moveWindow("Static camera warped", 1080 // 2 + 40, 1080 // 2 + 40)
        cv.moveWindow("Light direction", 1080 // 2 + 500, 1080 // 2 + 40)

    # Loop until static video is finished
    while static.isOpened() and moving.isOpened():
        # get static frame
        ret, frame_static = static.read()
        if not ret:
            break

        # get static milliseconds
        ms = static.get(cv.CAP_PROP_POS_MSEC)

        # calculate moving next frame
        moving.set(cv.CAP_PROP_POS_FRAMES, np.round((ms / 1000 + moving_delay) * moving_fps))

        # get moving frame
        ret, frame_moving = moving.read()
        if not ret:
            break

        # undistort moving
        frame_moving = cv.undistort(frame_moving, K, dist)

        # get square and circle from both cams
        square_static, circle_static = detect_square(frame_static)
        square_moving, circle_moving = detect_square(frame_moving)

        # calculate both homographies
        if not np.array_equal(square_static, SQUARE_NOT_FOUND) and not np.array_equal(
            square_moving, SQUARE_NOT_FOUND
        ):
            # H1, _ = cv.findHomography(square_static, marker3d)
            Hs, _ = cv.findHomography(square_static, marker)
            Hm, _ = cv.findHomography(marker, square_moving)
            warped_static = cv.warpPerspective(frame_static, Hs, (500, 500))
            yuv = cv.cvtColor(warped_static, cv.COLOR_BGR2YUV)
            MLIC.append(yuv[0])
            U.append(yuv[1])
            V.append(yuv[2])

            RT = np.linalg.inv(K) @ Hm
            R1norm = cv.norm(RT[:, 0])
            R2norm = cv.norm(RT[:, 1])
            alpha = np.average((R1norm, R2norm))
            R1, R2, T = RT.T / alpha
            R = np.column_stack((R1, R2, np.cross(R1, R2)))
            RTinv = np.column_stack((R.T, -R.T @ T))

            camera_position = RTinv @ [0, 0, 0, 1]
            u, v, _ = camera_position / np.linalg.norm(camera_position)
            L.append((u, v))

        #### DEBUG SECTION BEGIN
        if debug:
            cv.imshow("Static camera", draw_debug(frame_static, square_static, circle_static))
            cv.imshow("Moving camera", draw_debug(frame_moving, square_moving, circle_moving))
            if warped_static is not None:
                cv.imshow("Static camera warped", warped_static)
                cv.imshow("Light direction", draw_light_direction(u, v))

            # Press Q on keyboard to exit
            if cv.waitKey(1) & 0xFF == ord("q"):
                static.release()
                moving.release()
                cv.destroyAllWindows()
                return
        #### DEBUG SECTION END

    # TODO: store warped image and (u,v) on .npz file
    np.savez(
        DEFAULT_OUTPUT_PATH / Path(moving_video_path).stem,
        mlic=MLIC,
        l=L,
        u=np.average(U, axis=0),
        v=np.average(V, axis=0),
    )

    static.release()
    moving.release()

    if debug:
        cv.destroyAllWindows()


def main():
    calibration_mtx_npy = Path(__file__).parent / "output/K.npy"
    distortion_mtx_npy = Path(__file__).parent / "output/dist.npy"
    static_video = Path(__file__).parent / f"data/cam1 - static/coin{NUM_VIDEO}.mov"
    moving_video = Path(__file__).parent / f"data/cam2 - moving light/coin{NUM_VIDEO}.mp4"

    analyse(static_video, moving_video, calibration_mtx_npy, distortion_mtx_npy, debug=FLG_DEBUG)


if __name__ == "__main__":
    main()
