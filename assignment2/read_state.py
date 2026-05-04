import rtde_control
import rtde_receive


def connect(ip: str):
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)

    return rtde_c, rtde_r


def main():
    ip = "192.168.1.103"
    rtde_c, rtde_r = connect(ip)

    pose = rtde_r.getActualTCPPose()
    joints = rtde_r.getActualQ()

    print("TCP pose:  ", pose)
    print("Joints:    ", joints)

    rtde_c.disconnect()
    rtde_r.disconnect()


if __name__ == "__main__":
    main()
