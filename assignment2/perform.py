import argparse
import time
import pandas as pd
import rtde_control
import rtde_receive

IP = "192.168.1.103"
SPEED = 0.05
ACCELERATION = 0.1
LOOKAHEAD = 0.1
GAIN = 300

# standard ready pose — arm in front, roughly horizontal
HOME_JOINTS = [0, -1.5708, 1.5708, 3.14, -1.5708, 0]


def load_trajectory(path, tcp):
    df = pd.read_csv(path)
    df["x"] += tcp[0]
    df["y"] += tcp[1]
    df["z"] += tcp[2]
    # fix orientation to home pose — ignore recorded mocap orientation
    df["rx"] = tcp[3]
    df["ry"] = tcp[4]
    df["rz"] = tcp[5]
    return df


def perform(rtde_c, df):
    timestamps = df["timestamp"].values
    poses = df[["x", "y", "z", "rx", "ry", "rz"]].values

    print(f"Executing {len(poses)} waypoints...")
    for i, (t, pose) in enumerate(zip(timestamps, poses)):
        dt = (
            timestamps[i] - timestamps[i - 1]
            if i > 0
            else timestamps[1] - timestamps[0]
        )
        rtde_c.servoL(list(pose), SPEED, ACCELERATION, dt, LOOKAHEAD, GAIN)
        time.sleep(dt)

    rtde_c.servoStop()
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="trajectory_ready.csv")
    args = parser.parse_args()

    rtde_c = rtde_control.RTDEControlInterface(IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(IP)

    print("Moving to home position...")
    rtde_c.moveJ(HOME_JOINTS, speed=0.5, acceleration=0.5)

    tcp = rtde_r.getActualTCPPose()
    print(f"Home TCP: {[round(v, 4) for v in tcp]}")

    df = load_trajectory(args.file, tcp)
    print(
        f"First waypoint: {[round(v, 4) for v in df.iloc[0][['x', 'y', 'z', 'rx', 'ry', 'rz']]]}"
    )
    print(
        f"Last  waypoint: {[round(v, 4) for v in df.iloc[-1][['x', 'y', 'z', 'rx', 'ry', 'rz']]]}"
    )
    input("Press Enter to start (make sure someone is at the e-stop)...")

    perform(rtde_c, df)

    rtde_c.disconnect()
    rtde_r.disconnect()


if __name__ == "__main__":
    main()
