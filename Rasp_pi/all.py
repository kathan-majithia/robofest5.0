from dronekit import connect, VehicleMode
import termios, tty, sys, time

PORT = "/dev/ttyACM0"
BAUD = 115200

# -----------------------------
# Throttle (CH3)
# -----------------------------
THROTTLE_MIN = 1000
THROTTLE_MAX = 1150
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
# Pitch (CH2)
# -----------------------------
PITCH_CENTER = 1500
PITCH_MIN = 1400
PITCH_MAX = 1600
current_pitch = PITCH_CENTER
pitch_step = 1

# -----------------------------
# Roll (CH1)
# -----------------------------
ROLL_CENTER = 1500
ROLL_MIN = 1400
ROLL_MAX = 1600
current_roll = ROLL_CENTER
roll_step = 5

# -----------------------------
# Keyboard input
# -----------------------------
def pa(vehicle):
    alt = vehicle.location.global_relative_frame.alt
    if alt is not None:
        print(f"Altitude = {alt:.2f} m")
    else:
        print("Altitude = N/A")

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
    print(f"Throttle = {pwm}", end=" ")

def set_yaw(vehicle, pwm):
    vehicle.channels.overrides['4'] = pwm
    print(f"Yaw = {pwm}", end=" ")

def set_pitch(vehicle, pwm):
    vehicle.channels.overrides['2'] = pwm
    print(f"Pitch = {pwm}", end=" ")

def set_roll(vehicle, pwm):
    vehicle.channels.overrides['1'] = pwm
    print(f"Roll = {pwm}", end=" ")

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
    current_pitch = PITCH_CENTER
    current_roll = ROLL_CENTER
    
    set_throttle(vehicle, current_throttle)
    set_yaw(vehicle, current_yaw)
    set_pitch(vehicle, current_pitch)
    set_roll(vehicle, current_roll)
    
    print("\n=== CONTROLS ===")
    print("Numpad-style layout:")
    print("    7   8   9")
    print("    4   5   6")
    print("    1   2   3")
    print()
    print("W → Throttle +10 (up)")
    print("S → Throttle -10 (down)")
    print("1 → Yaw -5 (rotate left)")
    print("3 → Yaw +5 (rotate right)")
    print("4 → Roll -5 (strafe left)")
    print("6 → Roll +5 (strafe right)")
    print("7 → Pitch -1 (backward)")
    print("9 → Pitch +1 (forward)")
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
        
        # Roll
        elif key == "4":
            current_roll = max(current_roll - roll_step, ROLL_MIN)
            set_roll(vehicle, current_roll)
        elif key == "6":
            current_roll = min(current_roll + roll_step, ROLL_MAX)
            set_roll(vehicle, current_roll)
        
        # Pitch
        elif key == "7":
            current_pitch = max(current_pitch - pitch_step, PITCH_MIN)
            set_pitch(vehicle, current_pitch)
        elif key == "9":
            current_pitch = min(current_pitch + pitch_step, PITCH_MAX)
            set_pitch(vehicle, current_pitch)
        
        elif key.lower() == "q":
            print("\nExiting...")
            break
        
        print()  # newline after each command
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nEmergency stop!")

finally:
    vehicle.channels.overrides['3'] = THROTTLE_MIN
    vehicle.channels.overrides = {}
    force_disarm(vehicle)
    vehicle.close()
    print("Done.")