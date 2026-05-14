import rtde_control
import rtde_receive
import time

ROBOT_IP = "192.168.1.103"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

q = rtde_r.getActualQ()

print("Current q:", q)

# Tiny motion
q[0] += 0.02

rtde_c.moveJ(
    q,
    speed=0.05,
    acceleration=0.05
)

time.sleep(1)

print("New q:", rtde_r.getActualQ())

# Clean shutdown
rtde_c.disconnect()
rtde_r.disconnect()