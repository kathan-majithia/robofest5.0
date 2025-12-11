import time, sys, os
from dronekit import connect, VehicleMode

# CONFIG
CONN = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"   # or "udp:127.0.0.1:14550"
BAUD = 115200
THROTTLE_MIN = 1000       # safe idle
THROTTLE_TEST = 1200      # gentle spin (increase only with props removed)
TEST_SECONDS = 5

def confirm():
    print("!!! SAFETY WARNING !!!")
    print("Props must be REMOVED. Aircraft must be secured.")
    s = input("Type 'YES' (uppercase) to proceed with bench test: ")
    return s == "YES"

def main():
    if not confirm():
        print("User did not confirm. Exiting.")
        return

    print("Connecting to", CONN, "baud", BAUD)
    vehicle = connect(CONN, baud=BAUD, wait_ready=True, heartbeat_timeout=30, timeout=30)

    try:
        print("Connected: mode:", vehicle.mode.name, "armed:", vehicle.armed)
        print("Battery:", vehicle.battery)

        # Save original params to restore later
        orig_arm_check = vehicle.parameters.get('ARMING_CHECK', None)
        orig_safety = vehicle.parameters.get('BRD_SAFETYENABLE', None)
        print("Original ARMING_CHECK:", orig_arm_check, "BRD_SAFETYENABLE:", orig_safety)

        # Quick battery check
        try:
            vb = vehicle.battery.voltage
            print("Battery voltage:", vb)
            if vb is not None and vb < 7.0:
                print("Battery voltage too low for bench test:", vb)
                return
        except Exception:
            pass

        # Disable arming checks & safety (bench only)
        print("Setting ARMING_CHECK=0 and BRD_SAFETYENABLE=0 (bench only)")
        if orig_arm_check is not None:
            vehicle.parameters['ARMING_CHECK'] = 0
        else:
            print("Warning: could not read ARMING_CHECK; will try to set anyway.")
            vehicle.parameters['ARMING_CHECK'] = 0

        if orig_safety is not None:
            vehicle.parameters['BRD_SAFETYENABLE'] = 0
        else:
            vehicle.parameters['BRD_SAFETYENABLE'] = 0

        # short wait to let params write
        time.sleep(1.0)

        # Ensure a safe mode (STABILIZE)
        try:
            vehicle.mode = VehicleMode("STABILIZE")
            time.sleep(0.5)
        except Exception:
            pass

        # Arm
        print("Arming vehicle...")
        vehicle.armed = True
        t0 = time.time()
        while not vehicle.armed and (time.time() - t0 < 10):
            print("Waiting for arm...")
            time.sleep(0.5)
        if not vehicle.armed:
            print("Failed to arm. Aborting and restoring params.")
            return
        print("ARMED!")

        # Apply throttle override
        print("Applying throttle override:", THROTTLE_TEST)
        vehicle.channels.overrides['3'] = THROTTLE_TEST

        # run test
        for i in range(TEST_SECONDS):
            print(f"Testing... {i+1}/{TEST_SECONDS}")
            time.sleep(1)

        # restore throttle
        print("Restoring throttle to min and clearing overrides.")
        vehicle.channels.overrides['3'] = THROTTLE_MIN
        time.sleep(0.5)
        vehicle.channels.overrides = {}

        # Disarm
        print("Disarming...")
        vehicle.armed = False
        t0 = time.time()
        while vehicle.armed and (time.time() - t0 < 5):
            time.sleep(0.2)

        print("Disarmed.")

    finally:
        # Restore params if we saved them
        try:
            if orig_arm_check is not None:
                print("Restoring ARMING_CHECK to", orig_arm_check)
                vehicle.parameters['ARMING_CHECK'] = orig_arm_check
        except Exception as e:
            print("Warning: could not restore ARMING_CHECK:", e)

        try:
            if orig_safety is not None:
                print("Restoring BRD_SAFETYENABLE to", orig_safety)
                vehicle.parameters['BRD_SAFETYENABLE'] = orig_safety
        except Exception as e:
            print("Warning: could not restore BRD_SAFETYENABLE:", e)

        try:
            vehicle.channels.overrides = {}
        except Exception:
            pass
        try:
            vehicle.close()
        except Exception:
            pass
        print("Done. Remember to re-enable safety/arm checks for flight.")

if __name__ == "__main__":
    main()