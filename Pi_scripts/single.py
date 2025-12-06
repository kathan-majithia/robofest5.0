from pymavlink import mavutil
import time
import termios, sys, tty

# --------------------------
# Keyboard input function
# --------------------------
def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# --------------------------
# MAVLink connection
# --------------------------
master = mavutil.mavlink_connection('/dev/serial0', baud=57600)

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
# RC Override helper
# --------------------------
def send_single_motor(motor_num, throttle):
    """
    Motor mapping:
    ArduCopter RC override maps motors like this for PWM output:
    Motor 1 = CH3 (Throttle)    → we will isolate using SERVO functions
    But for direct override, we simulate one motor by only affecting its output channel.
    """

    # All 8 channels default to None (0 = no override)
    ch = [0,0,0,0,0,0,0,0]

    # Motor → RC channel mapping (for bench test)
    # NOTE: This works because ArduPilot routes channels to SERVO outputs.
    # Motor 1 = RC channel 3
    # Motor 2 = RC channel 4
    # Motor 3 = RC channel 5
    # Motor 4 = RC channel 6
    motor_channel = {
        1: 3,
        2: 4,
        3: 5,
        4: 6
    }

    rc_index = motor_channel[motor_num] - 1
    ch[rc_index] = throttle

    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        *ch
    )

# --------------------------
# Ask user which motor
# --------------------------
print("\nWhich motor do you want to test?")
print("1 = Motor 1")
print("2 = Motor 2")
print("3 = Motor 3")
print("4 = Motor 4")
motor = int(input("Enter motor number: "))

if motor not in [1,2,3,4]:
    print("Invalid motor number.")
    sys.exit()

# --------------------------
# Throttle config
# --------------------------
throttle = 1100
step = 2
minimum = 1100
maximum = 1500

print("\nArming motors...")
arm()
time.sleep(2)

print("\nControls:")
print("  W = +2 throttle")
print("  S = -2 throttle")
print("  Q = quit and disarm")
print(f"\nTesting MOTOR {motor}")
print(f"Throttle: {throttle}")

send_single_motor(motor, throttle)

# --------------------------
# Main loop
# --------------------------
while True:
    key = getch()

    if key.lower() == 'w':
        throttle += step
        if throttle > maximum:
            throttle = maximum
        print(f"Throttle ↑ : {throttle}")
        send_single_motor(motor, throttle)

    elif key.lower() == 's':
        throttle -= step
        if throttle < minimum:
            throttle = minimum
        print(f"Throttle ↓ : {throttle}")
        send_single_motor(motor, throttle)

    elif key.lower() == 'q':
        print("\nStopping motor…")
        send_single_motor(motor, 1000)
        time.sleep(1)
        print("Disarming…")
        disarm()
        print("Done.")
        break

    time.sleep(0.05)
