import argparse
import warnings
import numpy as np
import cv2 as cv
import librosa
from scipy import signal
from pathlib import Path

warnings.filterwarnings("ignore")  # Needed by Librosa for video sync


################### INTERNAL PARAMETERS ###################
SQUARE_DETECTION_BLUR_KERNEL = (5, 5)
SQUARE_DETECTION_BLUR_SIGMA = 1.5
SQUARE_DETECTION_CONTOURS_EPS = 0.005
SQUARE_NOT_FOUND = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
CIRCLE_NOT_FOUND = np.array([-1, -1])
DEFAULT_MARKER_DIM = 400

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
    square = square.astype(int)
    circle = circle.astype(int)
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


def get_videos_sync_time(video1_path: str | Path, video2_path: str | Path) -> tuple[int, int]:
    audio1, sr = librosa.load(Path(video1_path))
    audio2, sr = librosa.load(Path(video2_path))
    correlation = signal.correlate(audio1, audio2, mode="full")
    lags = signal.correlation_lags(audio1.size, audio2.size, mode="full")
    lag = lags[np.argmax(correlation)]

    if lag < 0:
        return 0, -lag / sr

    return lag / sr, 0


def build_marker(dim: int):
    return np.array([[0, 0, 1], [dim, 0, 1], [dim, dim, 1], [0, dim, 1]])


def detect_square(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, SQUARE_DETECTION_BLUR_KERNEL, SQUARE_DETECTION_BLUR_SIGMA)

    # highlight black features (the square) in order to attenuate the shadow
    # NB: tophat flips blacks and whites in the image
    se = cv.getStructuringElement(cv.MORPH_RECT, (200, 200))
    bh = cv.morphologyEx(blur, cv.MORPH_BLACKHAT, se)

    _, thresh = cv.threshold(bh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Hierarchical closed contour extraction
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Ramer-Douglas-Peucker algorithm for contour approximation
    approx = [cv.approxPolyDP(c, SQUARE_DETECTION_CONTOURS_EPS * cv.arcLength(c, True), True) for c in contours]

    # Keep only contours with four corners (possible squares)
    squares = [a.squeeze() for a in approx if len(a) == 4]

    if len(squares) < 2:
        return SQUARE_NOT_FOUND, CIRCLE_NOT_FOUND

    # Sort by area in order to identify the two biggest ones
    [internal_square, external_square] = sorted(squares, key=cv.contourArea)[-2:]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    internal_square = cv.cornerSubPix(thresh, internal_square.astype("float32"), (11, 11), (-1, -1), criteria)
    external_square = cv.cornerSubPix(thresh, external_square.astype("float32"), (11, 11), (-1, -1), criteria)

    # STEP 1. Align internal and external squares corners
    # Given the findContour algorithm implemented by opencv, we assume that
    #   the corners of the external square are extracted counter-clockwise,
    #   while the corners of the internal square are extracted clockwise,
    #   beginning with the corner in the top-left.
    # Source: Paper 'Topological Structural Analysis of Digitized Binary Images by Border Following', Appendix 1
    # Useful: https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation

    # STEP 2. Find corners pair near to the white circle
    circles_found: int = 0
    for idx in range(4):
        # Circle is not exactly midway between the two corners
        # Displacement values computed from the marker.svg, for better debug visualization
        # true_midpoint = np.round((internal_square[idx] + external_square[(4-idx)%4]) / 2).astype(int)
        i = internal_square[idx]
        e = external_square[(4 - idx) % 4]
        midpoint = i + (e - i) * [0.4329842, 0.460332]

        # Check if midpoint is black (because colors were inverted by blackhat)
        if thresh[*midpoint.astype(int)[::-1]] == 0:
            circle = midpoint
            circle_pos = idx
            circles_found += 1

    if circles_found != 1:
        return SQUARE_NOT_FOUND, CIRCLE_NOT_FOUND

    # STEP 3. Sort internal square corners accordingly
    internal_square = np.roll(internal_square, -circle_pos, axis=0)

    return internal_square, circle


def analyse(
    static_video_path: str | Path,
    moving_video_path: str | Path,
    calibration_matrix_npy: str | Path,
    distortion_matrix_npy: str | Path,
    output_path: str | Path = None,
    marker_dim=DEFAULT_MARKER_DIM,
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

    # Sync videos by audio: set starting frames
    static_start_time, moving_start_time = get_videos_sync_time(static_video_path, moving_video_path)
    delay = moving_start_time - static_start_time
    static_frame_pos = np.round(static_start_time * static_fps)
    moving_frame_pos = np.round(moving_start_time * moving_fps)
    static.set(cv.CAP_PROP_POS_FRAMES, static_frame_pos)
    moving.set(cv.CAP_PROP_POS_FRAMES, moving_frame_pos)

    marker = build_marker(marker_dim)

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
        ret1, frame_static = static.read()
        ret2, frame_moving = moving.read()
        if not ret1 or not ret2:
            break

        # undistort moving
        frame_moving = cv.undistort(frame_moving, K, dist)

        # get square and circle from both cams
        square_static, circle_static = detect_square(frame_static)
        square_moving, circle_moving = detect_square(frame_moving)

        # calculate both homographies
        if not np.array_equal(square_static, SQUARE_NOT_FOUND) and not np.array_equal(square_moving, SQUARE_NOT_FOUND):
            Hs, _ = cv.findHomography(square_static, marker)
            Hm, _ = cv.findHomography(marker, square_moving)
            warped_static = cv.warpPerspective(frame_static, Hs, (marker_dim, marker_dim))
            yuv = cv.cvtColor(warped_static, cv.COLOR_BGR2YUV)

            if Hm is not None and Hm.size > 0:
                RT = np.linalg.inv(K) @ Hm
                R1norm = cv.norm(RT[:, 0])
                R2norm = cv.norm(RT[:, 1])
                alpha = np.average((R1norm, R2norm))
                R1, R2, T = RT.T / alpha
                R = np.column_stack((R1, R2, np.cross(R1, R2)))
                RTinv = np.column_stack((R.T, -R.T @ T))

                camera_position = RTinv @ [0, 0, 0, 1]
                u, v, _ = camera_position / np.linalg.norm(camera_position)

                MLIC.append(yuv[..., 0])
                U.append(yuv[..., 1])
                V.append(yuv[..., 2])
                L.append((u, v))

        # Keep track of the frames and skip frames if needed
        static_frame_pos += 1
        moving_frame_pos += 1
        static_time_pos = static_frame_pos / static_fps
        moving_time_pos = moving_frame_pos / moving_fps

        # Check if we need to skip frames in moving video
        moving_frame_pos_tobe = np.round((static_time_pos + delay) * moving_fps)
        if moving_frame_pos_tobe > moving_frame_pos + 1:
            moving_frame_pos = moving_frame_pos_tobe
            moving.set(cv.CAP_PROP_POS_FRAMES, moving_frame_pos)
            print("Frame skipped in moving video")

        # Check if we need to skip frames in static video
        static_frame_pos_tobe = np.round((moving_time_pos - delay) * static_fps)
        if static_frame_pos_tobe > static_frame_pos:
            static_frame_pos = static_frame_pos_tobe
            static.set(cv.CAP_PROP_POS_FRAMES, static_frame_pos)
            print("Frame skipped in static video")

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

    # Ensure output paths exists
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Store warped image and (u,v) on .npz file
    np.savez(
        Path(output_path) if output_path else DEFAULT_OUTPUT_PATH / Path(moving_video_path).stem,
        mlic=MLIC,
        l=L,
        u=np.average(U, axis=0),
        v=np.average(V, axis=0),
    )

    static.release()
    moving.release()

    if debug:
        cv.destroyAllWindows()


def init_cli():
    parser = argparse.ArgumentParser(description="RTI videos analysis")
    parser.add_argument("static_path", type=Path, help="Static video file")
    parser.add_argument("moving_path", type=Path, help="Moving video file")
    parser.add_argument("calibration_path", type=Path, help="Calibration matrix ouput path")
    parser.add_argument("distortion_path", type=Path, help="Distortion matrix output path")
    parser.add_argument(
        "-m", "--marker_dimension", type=int, help="Marker scale dimension", default=DEFAULT_MARKER_DIM
    )
    parser.add_argument("-o", "--output-path", type=Path, help="Output path", default=None)
    parser.add_argument("-d", "--debug", type=str, action=argparse.BooleanOptionalAction)
    return parser


def main():
    args = init_cli().parse_args()

    analyse(
        static_video_path=args.static_path,
        moving_video_path=args.moving_path,
        calibration_matrix_npy=args.calibration_path,
        distortion_matrix_npy=args.distortion_path,
        output_path=args.output_path,
        marker_dim=args.marker_dimension,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
