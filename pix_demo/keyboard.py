import time
import sys
import termios
import tty
from dronekit import connect, VehicleMode

# -----------------------------
# Keyboard helper (non-blocking)
# -----------------------------
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# -----------------------------
# Connect to Pixhawk
# -----------------------------
connection = "/dev/ttyACM0"
baudrate = 115200

print(f"Connecting to Pixhawk on {connection}...")
vehicle = connect(connection, baud=baudrate, wait_ready=True, timeout=30)
print("Connected!")
print("Mode:", vehicle.mode.name, "Armed:", vehicle.armed)

# -----------------------------
# Arm vehicle safely
# -----------------------------
print("\nArming motors...")
vehicle.mode = VehicleMode("STABILIZE")
vehicle.armed = True

t0 = time.time()
while not vehicle.armed:
    print(" Waiting for arm...")
    time.sleep(0.5)
    if time.time() - t0 > 15:
        print("ERROR: Could not arm!")
        exit()

print("✔ Armed!")

# -----------------------------
# Throttle control
# -----------------------------
throttle = 1000
vehicle.channels.overrides['3'] = throttle

print("""
------------------------------------------
Throttle Control Active
    W = increase throttle (+10)
    S = decrease throttle (-10)
    Q = ramp down + disarm + quit
------------------------------------------
Current Throttle = 1000
""")

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        key = getch().lower()

        if key == 'w':
            throttle = min(2000, throttle + 10)
            vehicle.channels.overrides['3'] = throttle
            print("Throttle:", throttle)

        elif key == 's':
            throttle = max(1000, throttle - 10)
            vehicle.channels.overrides['3'] = throttle
            print("Throttle:", throttle)

        elif key == 'q':
            print("\nRamping down and disarming...")
            break

        # keep override alive
        vehicle.channels.overrides['3'] = throttle
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

# -----------------------------
# Safe shutdown
# -----------------------------
print("→ Ramping down throttle...")
for pwm in range(throttle, 999, -20):
    vehicle.channels.overrides['3'] = pwm
    print("Throttle:", pwm)
    time.sleep(0.1)

vehicle.channels.overrides = {}
time.sleep(0.3)

vehicle.armed = False
time.sleep(1)

print("✔ Disarmed, exiting cleanly.")
vehicle.close()