import argparse
import sys
import pandas as pd

AXES = ["x", "y", "z"]


def center(df):
    for ax in AXES:
        df[ax] -= df[ax].mean()
    return df


def scale(df):
    max_range = max(df[ax].max() - df[ax].min() for ax in AXES)
    for ax in AXES:
        df[ax] /= max_range
    return df


def flatten_y(df):
    df["y"] = 0.0
    return df


def normalize(df):
    return scale(center(flatten_y(df)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="trajectory.csv")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    result = normalize(df)
    for ax in AXES:
        result[ax] *= args.scale
    result.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
