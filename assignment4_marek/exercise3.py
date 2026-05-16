import time

import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import rtde_control
import rtde_receive
from pynput import keyboard

from exercise2 import UR5E, UR5E_XML

ROBOT_IP = "192.168.1.103"
STEP_SIZE = 0.005  # metres per loop tick (5 mm)

KEY_TO_DELTA = {
    "w": (0, +1),
    "s": (0, -1),
    "a": (1, +1),
    "d": (1, -1),
    "q": (2, +1),
    "e": (2, -1),
}


class RobotConnection:
    def __init__(self, ip: str):
        self.ip = ip
        self.c: rtde_control.RTDEControlInterface | None = None
        self.r: rtde_receive.RTDEReceiveInterface | None = None

    def __enter__(self):
        print(f"[robot] connecting to {self.ip}...")
        self.c = rtde_control.RTDEControlInterface(self.ip)
        self.r = rtde_receive.RTDEReceiveInterface(self.ip)
        print("[robot] connected")
        return self

    def __exit__(self, *exc):
        print("[robot] disconnecting...")
        if self.c is not None:
            self.c.servoStop()
            self.c.stopScript()
            self.c.disconnect()
        if self.r is not None:
            self.r.disconnect()
        print("[robot] disconnected")

    def get_q(self):
        assert self.r is not None
        return jnp.array(self.r.getActualQ())

    def servo_j(self, q, velocity=0.5, accel=0.5, dt=1 / 125, lookahead=0.1, gain=300):
        assert self.c is not None
        self.c.servoJ(list(q), velocity, accel, dt, lookahead, gain)


class KeyboardControl:
    def __init__(self, key_to_delta: dict[str, tuple[int, int]], step_size: float):
        self.key_to_delta = key_to_delta
        self.step_size = step_size
        self.keys_pressed: set[str] = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    def __enter__(self):
        self.listener.start()
        return self

    def __exit__(self, *exc):
        self.listener.stop()

    @property
    def running(self) -> bool:
        return self.listener.running

    def delta(self):
        move = np.zeros(3)
        for k, (axis, sign) in self.key_to_delta.items():
            if k in self.keys_pressed:
                move[axis] += sign * self.step_size
        return jnp.array(move)

    def _on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            return False


ur5e = UR5E.from_xml(UR5E_XML)
with (
    RobotConnection(ROBOT_IP) as robot,
    KeyboardControl(KEY_TO_DELTA, STEP_SIZE) as jog,
):
    q = robot.get_q()
    print(f"[init] starting q (rad): {np.array(q).round(3).tolist()}")
    ur5e.sync(q)
    target_pos, target_rot = ur5e.forward(q)

    print("[viewer] launching - jog with W/A/S/D/Q/E, Esc to quit")
    with mujoco.viewer.launch_passive(ur5e.mj_model, ur5e.mj_data) as viewer:
        while viewer.is_running() and jog.running:
            target_pos = target_pos + jog.delta()
            q = ur5e.compute_target_q(target_pos, target_rot, q)

            ur5e.sync(q)
            viewer.sync()

            robot.servo_j(q)
            time.sleep(0.01)
