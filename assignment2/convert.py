import argparse
import sys
import pandas as pd
from scipy.spatial.transform import Rotation


def quat_to_rotvec(df):
    rots = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values)
    rotvec = rots.as_rotvec()
    df = df.drop(columns=["qx", "qy", "qz", "qw"])
    df["rx"] = rotvec[:, 0]
    df["ry"] = rotvec[:, 1]
    df["rz"] = rotvec[:, 2]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="trajectory.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    quat_to_rotvec(df).to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
