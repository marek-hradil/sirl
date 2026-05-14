import time
from pathlib import Path

import mujoco
import mujoco.viewer
from mujoco import mjx

import jax
import jax.numpy as jnp

import numpy as np
import scipy.optimize as optim

from pynput import keyboard

# ==============================
# OPTIONAL: REAL UR5 CONNECTION
# ==============================

USE_REAL_ROBOT = True

if USE_REAL_ROBOT:
    import rtde_control
    import rtde_receive

    ROBOT_IP = "192.168.1.103"

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# ==============================
# LOAD MUJOCO MODEL
# ==============================

MODEL_PATH = Path(
    "universal_robots_ur5e/scene.xml"
)

mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# End-effector site
EE_SITE = mj_model.site("attachment_site").id

# ==============================
# FORWARD KINEMATICS
# ==============================

def forward_kinematics(q):

    data = mjx_data.replace(qpos=q)

    data = mjx.fwd_position(mjx_model, data)

    pos = data.site_xpos[EE_SITE]
    rot = data.site_xmat[EE_SITE].reshape(3, 3)

    return pos, rot

# ==============================
# POSE ERROR FUNCTION
# ==============================

def rotation_error(R1, R2):

    # Frobenius norm
    return jnp.sum((R1 - R2) ** 2)

def pose_error(q, target_pos, target_rot):

    pos, rot = forward_kinematics(q)

    pos_err = jnp.sum((pos - target_pos) ** 2)

    rot_err = rotation_error(rot, target_rot)

    return pos_err + 0.1 * rot_err

# JIT compile
jit_err = jax.jit(pose_error)

# Automatic differentiation
jit_grad = jax.jit(jax.grad(pose_error, argnums=0))

# ==============================
# IK SOLVER
# ==============================

def solve_ik(target_pos, target_rot, init_q):

    result = optim.minimize(
        fun=lambda q: np.array(
            jit_err(q, target_pos, target_rot),
            dtype=np.float64,
        ),
        x0=np.array(init_q),
        jac=lambda q: np.array(
            jit_grad(q, target_pos, target_rot),
            dtype=np.float64,
        ),
        method="L-BFGS-B",
        options={
            "maxiter": 100,
        },
    )

    return jnp.array(result.x)

# ==============================
# INITIAL ROBOT STATE
# ==============================

if USE_REAL_ROBOT:
    q = jnp.array(rtde_r.getActualQ())

else:
    q = jnp.deg2rad(
        jnp.array([-90, -60, 90, -30, 0, 0])
    )
target_pos, target_rot = forward_kinematics(q)

STEP_SIZE = 0.03

# ==============================
# KEYBOARD CONTROL
# ==============================

keys_pressed = set()

def on_press(key):

    try:
        keys_pressed.add(key.char)
    except:
        pass

def on_release(key):

    try:
        keys_pressed.remove(key.char)
    except:
        pass

    if key == keyboard.Key.esc:
        return False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release,
)

listener.start()

# ==============================
# MAIN LOOP
# ==============================

with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:

    while viewer.is_running():

        move = np.zeros(3)

        # Forward/back
        if 'w' in keys_pressed:
            move[0] += STEP_SIZE

        if 's' in keys_pressed:
            move[0] -= STEP_SIZE

        # Left/right
        if 'a' in keys_pressed:
            move[1] += STEP_SIZE

        if 'd' in keys_pressed:
            move[1] -= STEP_SIZE

        # Up/down
        if 'q' in keys_pressed:
            move[2] += STEP_SIZE

        if 'e' in keys_pressed:
            move[2] -= STEP_SIZE

        # Update target
        target_pos = target_pos + jnp.array(move)

        # Solve IK
        q = solve_ik(
            target_pos,
            target_rot,
            q,
        )

        # Apply to MuJoCo
        mj_data.qpos[:6] = np.array(q)

        mujoco.mj_step(mj_model, mj_data)

        viewer.sync()

        # ==========================
        # SEND TO REAL ROBOT
        # ==========================

        if USE_REAL_ROBOT:

            rtde_c.servoJ(
                np.array(q).tolist(),
                0.5,      # velocity
                0.5,      # acceleration
                1/125,    # dt
                0.1,      # lookahead_time
                300       # gain
            )

        time.sleep(0.01)

# ==============================
# CLEANUP
# ==============================

if USE_REAL_ROBOT:
    rtde_c.stopScript()