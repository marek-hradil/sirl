import mujoco
import mujoco.viewer
from mujoco import mjx
import time
from pathlib import Path
import jax
import jax.numpy as jnp
from pynput import keyboard
import scipy.optimize as optim
from pathlib import Path
# Path to your XML scene file
scene_path = Path('universal_robots_ur5e/scene.xml')

# Load the MuJoCo model
mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
mj_data = mujoco.MjData(mj_model)

# Transfer the MuJoCo model and data to MJX for hardware-accelerated simulation (GPU/TPU)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# This function answers:If the joints are q, where is the robot hand?
# this code is copied from assignment

def forward_kinematics(q):

    new_mjx_data = mjx_data.replace(qpos=q)

    new_mjx_data = mjx.fwd_position(
        mjx_model,
        new_mjx_data
    )

    pos = new_mjx_data.site_xpos[
        mj_model.site('attachment_site').id
    ]

    mat = new_mjx_data.site_xmat[
        mj_model.site('attachment_site').id
    ]

    return pos, mat

# Build the error fucntion
def pose_err(q, target_pos, target_mat):

    pos, mat = forward_kinematics(q)

    pos_err = jnp.sum((pos - target_pos)**2)

    rot_err = jnp.sum((mat - target_mat)**2)

    return pos_err + rot_err

#jit compilation to make the run much faster
jit_err = jax.jit(pose_err)
jit_err_grad = jax.jit(jax.grad(pose_err, argnums=(0,)))
def ik_optim(target_pos, target_mat, init_guess):
    result = optim.minimize(jit_err,args=(target_pos, target_mat),x0=init_guess,
                            method='L-BFGS-B',jac=jit_err_grad)
    return result.x





# Initial joint angles in degrees → converted to radians
initial_angles_deg = jnp.array([-90, -60, 90.2, -30, 0, 0])

initial_angles_rad = jnp.deg2rad(initial_angles_deg)

# Set initial control and qpos to the desired starting pose
for i in range(6):
    mj_data.ctrl[i] = float(initial_angles_rad[i])
    mj_data.qpos[i] = float(initial_angles_rad[i])

# Forward kinematics to apply initial pose
mujoco.mj_forward(mj_model, mj_data)

# Shared variable to store the current command
key_commands = {"w": False, "s": False, "a": False, "d": False,
                "q": False, "e": False, "r": False, "f": False,
                "t": False, "g": False, "y": False, "h": False}

def on_press(key):
    try:
        if key.char in key_commands:
            key_commands[key.char] = True
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char in key_commands:
            key_commands[key.char] = False
    except AttributeError:
        pass

# Start the keyboard listener in a background thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Controls:")
print("  W/S → Joint 0 (base rotate)")
print("  A/D → Joint 1 (shoulder)")
print("  Q/E → Joint 2 (elbow)")
print("  R/F → Joint 3 (wrist 1)")
print("  T/G → Joint 4 (wrist 2)")
print("  Y/H → Joint 5 (wrist 3)")

# Launch interactive viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    print("Viewer started. Close the window to exit.")
    while viewer.is_running():
        step_size = 0.01

        # Joint 0 — base
        if key_commands["w"]:
            #mj_data.ctrl[0] += step_size
            # init pos
            pos, rot = forward_kinematics(initial_angles_rad)
            ## move 0.2 meters
            target_pos = pos + jnp.array([0, 0, 0.3])
            target_rot = rot
            # run the optimation
            target_q = ik_optim(target_pos, target_rot, init_guess=initial_angles_rad)
            # Set initial control and qpos to the desired starting pose
            initial_angles_rad = target_q
            for i in range(6):
                mj_data.ctrl[i] = float(initial_angles_rad[i])
                mj_data.qpos[i] = float(initial_angles_rad[i])
        if key_commands["s"]:
            mj_data.ctrl[0] -= step_size

        # Joint 1 — shoulder
        if key_commands["a"]:
            mj_data.ctrl[1] += step_size
        if key_commands["d"]:
            mj_data.ctrl[1] -= step_size

        # Joint 2 — elbow
        if key_commands["q"]:
            mj_data.ctrl[2] += step_size
        if key_commands["e"]:
            mj_data.ctrl[2] -= step_size

        # Joint 3 — wrist 1
        if key_commands["r"]:
            mj_data.ctrl[3] += step_size
        if key_commands["f"]:
            mj_data.ctrl[3] -= step_size

        # Joint 4 — wrist 2
        if key_commands["t"]:
            mj_data.ctrl[4] += step_size
        if key_commands["g"]:
            mj_data.ctrl[4] -= step_size

        # Joint 5 — wrist 3
        if key_commands["y"]:
            mj_data.ctrl[5] += step_size
        if key_commands["h"]:
            mj_data.ctrl[5] -= step_size

        # Step the simulation
        mujoco.mj_step(mj_model, mj_data)

        # Sync viewer with simulation
        viewer.sync()

        # Maintain real-time pace
        time.sleep(mj_model.opt.timestep)

# Stop the listener when the viewer closes
listener.stop()