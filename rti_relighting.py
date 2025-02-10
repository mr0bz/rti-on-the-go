import argparse
import cv2 as cv
import numpy as np
from pathlib import Path


LIGHT_RADIUS = 200


def init_cli():
    parser = argparse.ArgumentParser(description="Relight object.")
    parser.add_argument("filename", type=Path, help="Path to the .npz RTI function file.")
    return parser


def main():
    args = init_cli().parse_args()
    data = np.load(args.filename, "r")

    F = data["f"]
    U = data["u"]
    V = data["v"]

    l_size, _, img_size, _ = F.shape

    lx = ly = lxg = lyg = 0
    while True:

        def mouse_callback(event, x, y, falgs, param):
            nonlocal lx, ly, lxg, lyg
            lxg = x
            lyg = y
            lx = x / LIGHT_RADIUS - 1.0
            ly = y / LIGHT_RADIUS - 1.0

        light = np.zeros((LIGHT_RADIUS * 2, LIGHT_RADIUS * 2), dtype=np.uint8)
        cv.circle(light, (LIGHT_RADIUS, LIGHT_RADIUS), LIGHT_RADIUS, (255, 255, 255), 2, cv.LINE_AA)
        cv.line(light, (LIGHT_RADIUS, LIGHT_RADIUS), (lxg, lyg), (255, 255, 255), 2, cv.LINE_AA)
        cv.circle(light, (lxg, lyg), 20, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Light direction", light)
        cv.setMouseCallback("Light direction", mouse_callback)

        lxi = int((lx + 1.0) / 2 * l_size)
        lyi = int((ly + 1.0) / 2 * l_size)

        yuv = np.zeros((img_size, img_size, 3))
        yuv[:, :, 0] = F[lyi, lxi]
        yuv[:, :, 1] = U
        yuv[:, :, 2] = V

        rgb = cv.cvtColor(yuv.astype(np.uint8), cv.COLOR_YUV2BGR)

        cv.imshow("Image", rgb)

        # Press Q on keyboard to exit
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            return


if __name__ == "__main__":
    main()
