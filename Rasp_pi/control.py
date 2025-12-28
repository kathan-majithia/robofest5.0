from dronekit import connect, VehicleMode
import termios, tty, sys, time

PORT = "/dev/ttyACM0"
BAUD = 115200

# Throttle settings
THROTTLE_MIN = 1000
THROTTLE_MAX = 1500
THROTTLE_STEP = 25

current_throttle = THROTTLE_MIN


# -----------------------------
# Keyboard input
# -----------------------------
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
# FORCE ARM Pixhawk
# -----------------------------
def force_arm(vehicle):
    print("\n=== FORCE ARMING ===")
    vehicle.mode = VehicleMode("STABILIZE")
    time.sleep(1)

    vehicle._master.mav.command_long_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        400,      # MAV_CMD_COMPONENT_ARM_DISARM
        0,
        1,        # ARM
        21196,    # MAGIC CODE to bypass checks
        0, 0, 0, 0, 0
    )

    while not vehicle.armed:
        print("  Waiting for forced arm...")
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
        0,       # DISARM
        21196,
        0, 0, 0, 0, 0
    )
    time.sleep(1)
    print("Vehicle DISARMED.")


# -----------------------------
# Set throttle
# -----------------------------
def set_throttle(vehicle, pwm):
    vehicle.channels.overrides['3'] = pwm
    print(f"Throttle = {pwm}")


# -----------------------------
# MAIN PROGRAM
# -----------------------------
print(f"Connecting to Pixhawk on {PORT}...")
vehicle = connect(PORT, baud=BAUD, wait_ready=False)
print("Connected.")

try:
    force_arm(vehicle)

    global current_throttle
    current_throttle = THROTTLE_MIN
    set_throttle(vehicle, current_throttle)

    print("\n=== CONTROL ===")
    print("W → Increase throttle")
    print("S → Decrease throttle")
    print("Q → Quit + Disarm")
    print("=================\n")

    while True:
        key = getch().lower()

        if key == "w":
            current_throttle += THROTTLE_STEP
            if current_throttle > THROTTLE_MAX:
                current_throttle = THROTTLE_MAX
            set_throttle(vehicle, current_throttle)

        elif key == "s":
            current_throttle -= THROTTLE_STEP
            if current_throttle < THROTTLE_MIN:
                current_throttle = THROTTLE_MIN
            set_throttle(vehicle, current_throttle)

        elif key == "q":
            print("\nExiting...")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nEmergency stop!")

finally:
    # Motor stop
    vehicle.channels.overrides['3'] = THROTTLE_MIN
    time.sleep(1)
    vehicle.channels.overrides = {}

    force_disarm(vehicle)

    vehicle.close()
    print("Done.")