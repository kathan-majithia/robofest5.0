from pymavlink import mavutil
import time
import termios, sys, tty
import signal

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

def disarm(force=False):
    # param1 = 0 (disarm)
    # param2 = 21196 enables "force" disarm on ArduCopter
    param2 = 21196 if force else 0
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, param2, 0, 0, 0, 0, 0
    )

# --------------------------
# RC Override helpers
# --------------------------
def send_rc_throttle(throttle):
    """
    Only override channel 3 (throttle).
    Other channels = 0 => 'no change' for them.
    """
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        0, 0, throttle, 0,
        0, 0, 0, 0
    )

def clear_rc_override():
    """
    Release RC override on all 8 channels by sending UINT16_MAX (65535).
    This tells ArduPilot to go back to normal RC input.
    """
    for _ in range(5):
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            65535, 65535, 65535, 65535,
            65535, 65535, 65535, 65535
        )
        time.sleep(0.05)

# --------------------------
# Safe shutdown
# --------------------------
def safe_shutdown():
    print("\n[SAFE SHUTDOWN] Sending low throttle & disarm...")

    # 1) spam low throttle a bit so FC definitely sees it
    for _ in range(20):
        send_rc_throttle(1000)   # your MINIMUM
        time.sleep(0.05)

    # 2) clear the override completely
    clear_rc_override()

    # 3) force disarm (works even if FC thinks it is flying)
    disarm(force=True)
    time.sleep(1.0)

    print("[SAFE SHUTDOWN] Override cleared and disarm sent.")

# Handle Ctrl+C as well
def sigint_handler(sig, frame):
    safe_shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

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
print("  W = Throttle +10")
print("  S = Throttle -10")
print("  Q = Quit and SAFE shutdown\n")

send_rc_throttle(throttle)

try:
    while True:
        key = getch()

        if key.lower() == 'w':
            throttle += step
            if throttle > maximum:
                throttle = maximum
            print(f"Throttle ↑ : {throttle}")
            send_rc_throttle(throttle)

        elif key.lower() == 's':
            throttle -= step
            if throttle < minimum:
                throttle = minimum
            print(f"Throttle ↓ : {throttle}")
            send_rc_throttle(throttle)

        elif key.lower() == 'q':
            print("Disarming and exiting...")
            safe_shutdown()
            break

        time.sleep(0.05)

finally:
    # In case of any unexpected error, still try to stop the motors
    safe_shutdown()
