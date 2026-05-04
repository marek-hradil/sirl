import argparse
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

POS_AXES = ["x", "y", "z"]
QUAT_COLS = ["qx", "qy", "qz", "qw"]


def resample_positions(t, df, t_new):
    return {ax: CubicSpline(t, df[ax].values)(t_new) for ax in POS_AXES}


def resample_quaternions(t, df, t_new):
    rots = Rotation.from_quat(df[QUAT_COLS].values)
    slerp = Slerp(t, rots)
    q = slerp(t_new).as_quat()
    return {col: q[:, i] for i, col in enumerate(QUAT_COLS)}


def resample(df, n_points):
    t = df["timestamp"].values
    t_new = np.linspace(t[0], t[-1], num=n_points)
    result = {"timestamp": t_new, "name": df["name"].iloc[0]}
    result.update(resample_positions(t, df, t_new))
    result.update(resample_quaternions(t, df, t_new))
    return pd.DataFrame(result)


def swap_xy(df):
    df["x"], df["y"] = df["y"].copy(), df["x"].copy()
    return df


def trim(df, n):
    return df.iloc[n:-n].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="trajectory.csv")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--trim", type=int, default=0)
    parser.add_argument("--swap-xy", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    if args.trim:
        df = trim(df, args.trim)
    if args.swap_xy:
        df = swap_xy(df)
    resample(df, args.points).to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
