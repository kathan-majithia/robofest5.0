from dronekit import connect, VehicleMode
import termios, tty, sys, time

PORT = "/dev/ttyACM0"
BAUD = 115200

# -----------------------------
# Throttle (CH3)
# -----------------------------
THROTTLE_MIN = 1000
THROTTLE_MAX = 1500
THROTTLE_STEP = 10
current_throttle = THROTTLE_MIN

# -----------------------------
# Yaw (CH4)
# -----------------------------
YAW_CENTER = 1340
YAW_MIN = 1200
YAW_MAX = 1750
current_yaw = YAW_CENTER
yaw_step = 5

# -----------------------------
# Keyboard input
# -----------------------------

def pa(vehicle):
    alt = vehicle.location.global_relative_frame.alt
    if alt is not None:
        print(f"Altitude = {alt:.2f} m"}
    else:
        printf("Altitude = N/A")

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# -----------------------------
# FORCE ARM
# -----------------------------
def force_arm(vehicle):
    print("\n=== FORCE ARMING ===")
    vehicle.mode = VehicleMode("STABILIZE")
    time.sleep(1)

    vehicle._master.mav.command_long_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        400,
        0,
        1,
        21196,
        0, 0, 0, 0, 0
    )

    while not vehicle.armed:
        print(" Waiting for forced arm...")
        time.sleep(0.5)

    print("Vehicle ARMED.\n")


# -----------------------------
# FORCE DISARM
# -----------------------------
def force_disarm(vehicle):
    print("\n=== FORCE DISARM ===")
    vehicle._master.mav.command_long_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        400,
        0,
        0,
        21196,
        0, 0, 0, 0, 0
    )
    time.sleep(1)
    print("Vehicle DISARMED.")


# -----------------------------
# RC Overrides
# -----------------------------
def set_throttle(vehicle, pwm):
    vehicle.channels.overrides['3'] = pwm
    print(f"Throttle = {pwm}",end="")

def set_yaw(vehicle, pwm):
    vehicle.channels.overrides['4'] = pwm
    print(f"Yaw = {pwm}",end="")


# -----------------------------
# MAIN
# -----------------------------
print(f"Connecting to Pixhawk on {PORT}...")
vehicle = connect(PORT, baud=BAUD, wait_ready=False)
print("Connected.")

try:
    force_arm(vehicle)

    current_throttle = THROTTLE_MIN
    current_yaw = YAW_CENTER

    set_throttle(vehicle, current_throttle)
    set_yaw(vehicle, current_yaw)

    print("\n=== CONTROLS ===")
    print("W → Throttle +25")
    print("S → Throttle -25")
    print("1 → Yaw -1")
    print("3 → Yaw +1")
    print("Q → Quit + Disarm")
    print("================\n")

    while True:
        key = getch()

        # Throttle
        if key.lower() == "w":
            current_throttle = min(current_throttle + THROTTLE_STEP, THROTTLE_MAX)
            set_throttle(vehicle, current_throttle)

        elif key.lower() == "s":
            current_throttle = max(current_throttle - THROTTLE_STEP, THROTTLE_MIN)
            set_throttle(vehicle, current_throttle)

        # Yaw
        elif key == "1":
            current_yaw = max(current_yaw - yaw_step, YAW_MIN)
            set_yaw(vehicle, current_yaw)

        elif key == "3":
            current_yaw = min(current_yaw + yaw_step, YAW_MAX)
            set_yaw(vehicle, current_yaw)

        elif key.lower() == "q":
            print("\nExiting...")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nEmergency stop!")

finally:
    vehicle.channels.overrides['3'] = THROTTLE_MIN
    vehicle.channels.overrides = {}
    force_disarm(vehicle)
    vehicle.close()
    print("Done.")