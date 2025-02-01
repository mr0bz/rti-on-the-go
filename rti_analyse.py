import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from pathlib import Path

############# DEBUGGING AND TESTING PARAMETERS ############
FLG_DEBUG = True
NUM_VIDEO = 2  # Choose test number (1 to 4)

################### INTERNAL PARAMETERS ###################
SQUARE_DETECTION_BLUR_KERNEL = (5, 5)
SQUARE_DETECTION_BLUR_SIGMA = 1.5
SQUARE_DETECTION_CONTOURS_EPS = 0.005
SQUARE_NOT_FOUND = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
CIRCLE_NOT_FOUND = np.array([-1, -1])
DEFAULT_MARKER = np.array([[0, 500], [0, 0], [500, 0], [500, 500]])
DEFAULT_MARKER_CIRCLE = np.array([-43.29842, 546.0332])

####################### DEBUG UTILS #######################
COLOR_BLUE = [255, 0, 0]
COLOR_CYAN = [255, 255, 0]
COLOR_RED = [0, 0, 255]
COLOR_GREEN = [0, 255, 0]


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


def calculate_rototranslation_matrix(H: np.matrix, K: np.matrix):
    RT = K ^ -1


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


def sync_videos(video1_path: str | Path, video2_path: str | Path) -> tuple[int, int]:
    return 0, 0


def analyse(
    static_video_path: str | Path,
    moving_video_path: str | Path,
    calibration_matrix_npy: str | Path,
    distortion_matrix_npy: str | Path,
    marker=DEFAULT_MARKER,
    debug=False,
):
    if debug:
        # Niceties for better visualization on 2K screen
        cv.namedWindow("Static camera", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow("Moving camera", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
        cv.namedWindow("Static camera warped", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)

        cv.resizeWindow("Static camera", 1080 // 2, 1920 // 2)
        cv.resizeWindow("Moving camera", 1920 // 2, 1080 // 2)
        cv.resizeWindow("Static camera warped", 375, 375)

        cv.moveWindow("Static camera", 20, 0)
        cv.moveWindow("Moving camera", 1080 // 2 + 40, 0)
        cv.moveWindow("Static camera warped", 1080 // 2 + 40, 1080 // 2 + 40)

    # Load calibration matrix
    Z = np.load(calibration_matrix_npy)
    dist = np.load(distortion_matrix_npy)

    # Get video from both cams
    static = cv.VideoCapture(static_video_path)
    moving = cv.VideoCapture(moving_video_path)

    # Get FPS for both videos, needed for sync loop
    static_fps = static.get(cv.CAP_PROP_FPS)
    moving_fps = moving.get(cv.CAP_PROP_FPS)

    # TODO: Sync videos by audio --> set starting frames
    # static_start, moving_start = get_videos_sync_time(static, moving)
    # static.set(cv.CAP_PROP_POS_FRAMES, np.round(static_start / static_fps))
    # moving.set(cv.CAP_PROP_POS_FRAMES, np.round(moving_start / moving_fps))

    # Loop until static video is finished
    n_frame = 0

    while static.isOpened() and moving.isOpened():
        # get static frame
        ret, frame_static = static.read()
        if not ret:
            break

        # get static milliseconds
        ms = static.get(cv.CAP_PROP_POS_MSEC)

        # calculate moving next frame
        moving.set(cv.CAP_PROP_POS_FRAMES, np.round(ms / 1000 * moving_fps))

        # get moving frame
        ret, frame_moving = moving.read()
        if not ret:
            break

        # undistort moving
        frame_moving = cv.undistort(frame_moving, Z, dist)

        # get square and circle from both cams
        square_static, circle_static = detect_square(frame_static)
        square_moving, circle_moving = detect_square(frame_moving)

        # calculate both homographies
        if not np.array_equal(square_static, SQUARE_NOT_FOUND) and not np.array_equal(
            square_moving, SQUARE_NOT_FOUND
        ):
            marker3d = np.insert(marker, 2, 0, 1)
            print(marker3d)
            H1, _ = cv.findHomography(square_static, marker3d)
            H2, _ = cv.findHomography(square_moving, marker)
            warped_static = cv.warpPerspective(frame_static, H1, (500, 500))

            M = np.linalg.inv(Z) @ H2
            R = M[:,:2]
            T = M[:,2]
            
            return

        # TODO: invert roto-translation matrix

        # TODO: calculate (u,v) coordinates

        # TODO: store warped image and (u,v) on list/array

        n_frame += 1

        #### DEBUG SECTION BEGIN
        if debug:
            cv.imshow("Static camera", draw_debug(frame_static, square_static, circle_static))
            cv.imshow("Moving camera", draw_debug(frame_moving, square_moving, circle_moving))
            if warped_static is not None:
                cv.imshow("Static camera warped", warped_static)

            # Press Q on keyboard to exit
            if cv.waitKey(1) & 0xFF == ord("q"):
                static.release()
                moving.release()
                cv.destroyAllWindows()
                return
        #### DEBUG SECTION END

    # TODO: store warped image and (u,v) on .npz file

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
