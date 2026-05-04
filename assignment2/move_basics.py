import rtde_control
import rtde_receive

IP = "192.168.1.103"

rtde_c = rtde_control.RTDEControlInterface(IP)
rtde_r = rtde_receive.RTDEReceiveInterface(IP)

# Read current pose
pose = rtde_r.getActualTCPPose()
print("Current pose:", [round(v, 4) for v in pose])

# Move 5 cm up in Z (index 2) from current position — safe first move
target = list(pose)
target[2] -= 0.1
print("Moving to:   ", [round(v, 4) for v in target])

rtde_c.moveL(target, speed=0.2, acceleration=0.1)

print("Done. New pose:", [round(v, 4) for v in rtde_r.getActualTCPPose()])

rtde_c.disconnect()
rtde_r.disconnect()
