import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import RBFInterpolator


DEFAULT_INTERP_METHOD = "RBF"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output"
GRID_SIZE = 32


def fit_model_rbf(l, vv, xx, yy):
    img_shape = vv.shape[-2:]
    vv = vv.reshape(vv.shape[0], -1)
    print(l.shape, vv.shape)
    rbfi = RBFInterpolator(l, vv, kernel="linear", smoothing=1)
    return rbfi(np.vstack((xx.flatten(), yy.flatten())).T).reshape((*xx.shape, *img_shape))


def fit_model_poly(l, vv, xx, yy):
    img_shape = vv.shape[-2:]
    [lx, ly] = l.T
    vv = vv.reshape(vv.shape[0], -1)
    A = np.vstack((lx**2, ly**2, lx * ly, lx, ly, np.ones((lx.shape[0])))).T
    b = vv
    coeffs = np.linalg.lstsq(A, b)[0]

    xxshape = xx.shape
    xx = xx.flatten()
    yy = yy.flatten()
    A = np.vstack((xx**2, yy**2, xx * yy, xx, yy, np.ones((xx.shape[0])))).T

    vinterp = A @ coeffs
    return np.reshape(vinterp, (*xxshape, *img_shape))


def init_cli():
    parser = argparse.ArgumentParser(description="Fit RTI Function.")
    parser.add_argument("filename", type=Path, help="Path to the .npz parameters file.")
    parser.add_argument(
        "-i",
        "--interpolation-method",
        type=str,
        help="Interpolation method.",
        default="RBF",
        choices=["RBF", "POLY"],
    )
    parser.add_argument("-g", "--grid-size", type=str, help="Granularity of light interpolation.", default=GRID_SIZE)
    return parser


def main():
    args = init_cli().parse_args()

    data_path = Path(args.filename)
    npz = np.load(data_path, "r")
    mlic: np.ndarray = npz["mlic"]
    l: np.ndarray = npz["l"]
    u: np.ndarray = npz["u"]
    v: np.ndarray = npz["v"]

    match args.interpolation_method:
        case "RBF":
            fit_model = fit_model_rbf
        case "POLY":
            fit_model = fit_model_poly
        case _:
            raise ValueError(f"Interpolation method '{args.interpolation_method}' is not valid.")

    steps = np.linspace(-1, 1, args.grid_size)
    xx = np.broadcast_to(steps, (args.grid_size, args.grid_size))
    yy = np.broadcast_to(steps, (args.grid_size, args.grid_size)).T

    F = fit_model(l, mlic, xx, yy)

    output_path = Path(
        f"/home/roberto/Code/cv/rti-on-the-go/output/F_{args.filename.stem}_{args.interpolation_method}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, f=F, u=u, v=v)


if __name__ == "__main__":
    main()
