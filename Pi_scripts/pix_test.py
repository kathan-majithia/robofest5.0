from pymavlink import mavutil
import time

# 1) Connect to FC over serial (same baud as SERIAL2_BAUD)
master = mavutil.mavlink_connection('/dev/serial0', baud=115200)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Heartbeat from system {master.target_system} component {master.target_component}")

# 2) Arm via MAVLink (ArduCopter)
def arm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )

def disarm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0
    )

# 3) Send RC overrides (1500 = mid, 1000 = min, 2000 = max)
#    channels: 1=roll, 2=pitch, 3=throttle, 4=yaw
def send_rc(roll=1500, pitch=1500, throttle=1000, yaw=1500):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        roll, pitch, throttle, yaw,
        0, 0, 0, 0  # channels 5-8 not overridden
    )

# ---- Test sequence ----
print("Arming...")
arm()
time.sleep(3)

print("Spinning motors slowly...")
for i in range(50):  # ~5 seconds at 10Hz
    send_rc(throttle=1800)  # just above idle
    time.sleep(0.1)

print("Stopping motors...")
for i in range(20):
    send_rc(throttle=1000)  # minimum
    time.sleep(0.1)

print("Disarming...")
disarm()
print("Done.")
