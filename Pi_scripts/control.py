from pymavlink import mavutil
import time
import termios, sys, tty

# --------------------------
# Keyboard input function
# --------------------------
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# --------------------------
# MAVLink connection
# --------------------------
master = mavutil.mavlink_connection('/dev/serial0', baud=115200)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to FC.")

# --------------------------
# Arm/Disarm functions
# --------------------------
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

# --------------------------
# RC Override
# --------------------------
def send_rc(throttle):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        0, 0, throttle, 0,   # ch1, ch2, ch3(throttle), ch4
        0, 0, 0, 0
    )

# --------------------------
# Main Control Logic
# --------------------------
throttle = 1000  # starting throttle
step = 10        # increment step
minimum = 1000
maximum = 1500

print("\nArming motors...")
arm()
time.sleep(2)

print("\nControls:")
print("  W = Throttle +50")
print("  S = Throttle -50")
print("  Q = Quit and Disarm\n")

send_rc(throttle)

# Loop for keyboard control
while True:
    key = getch()

    if key.lower() == 'w':
        throttle += step
        if throttle > maximum:
            throttle = maximum
        print(f"Throttle ↑ : {throttle}")
        send_rc(throttle)

    elif key.lower() == 's':
        throttle -= step
        if throttle < minimum:
            throttle = minimum
        print(f"Throttle ↓ : {throttle}")
        send_rc(throttle)

    elif key.lower() == 'q':
        print("Disarming and exiting...")
        send_rc(minimum)
        time.sleep(1)
        disarm()
        break

    time.sleep(0.05)
