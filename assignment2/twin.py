import time
import mujoco
import mujoco.viewer
import rtde_receive

ROBOT_IP  = "192.168.1.103"
MODEL_XML = "ur5e/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)

    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Connected to robot. Starting digital twin...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_read = time.perf_counter()
            q = rtde_r.getActualQ()
            t_recv = time.perf_counter()

            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            viewer.sync()

            latency_ms = (t_recv - t_read) * 1000
            print(f"\r  joints: {[round(v, 3) for v in q]}  read latency: {latency_ms:.1f} ms", end="", flush=True)

            time.sleep(0.01)  # ~100 Hz

    rtde_r.disconnect()


if __name__ == "__main__":
    main()
