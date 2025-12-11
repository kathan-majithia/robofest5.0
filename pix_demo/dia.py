import sys, time, traceback
from dronekit import connect, VehicleMode

# ----------------- CONFIG -----------------
DEFAULT_CONN = "/dev/ttyACM0"
DEFAULT_BAUD = 115200

TEST_THROTTLE = 1450   # target throttle PWM while testing (increase slowly if required)
RAMP_STEP = 25         # PWM increment per step
RAMP_DELAY = 0.6       # seconds between ramp increments
HOLD_SECONDS = 4       # hold at target for this many seconds
ARMABLE_WAIT = 30      # seconds to wait for vehicle.is_armable
PARAM_WRITE_RETRIES = 4
PARAM_WRITE_DELAY = 1.0
# ------------------------------------------

def confirm_user():
    print("\n!!! SAFETY WARNING !!!")
    print("1) PROPS MUST BE REMOVED.")
    print("2) AIRCRAFT MUST BE SECURE & CANNOT MOVE.")
    ans = input("Type EXACTLY 'YES' (uppercase) to proceed: ")
    return ans.strip() == "YES"

def try_set_param(vehicle, name, value, retries=PARAM_WRITE_RETRIES, delay=PARAM_WRITE_DELAY):
    """Try to set a parameter and verify it was written. Returns True on success."""
    for attempt in range(1, retries+1):
        try:
            vehicle.parameters[name] = value
        except Exception as e:
            # setting may fail transiently; continue to retry
            print(f"Warning: param set {name} attempt {attempt} failed: {e}")
        # small wait then verify
        time.sleep(delay)
        try:
            got = vehicle.parameters.get(name)
            # some parameters can be float-like or strings; use float compare if possible
            try:
                if got is None:
                    ok = False
                else:
                    ok = float(got) == float(value)
            except Exception:
                ok = (got == value)
            if ok:
                print(f"Param {name} -> {value} (confirmed)")
                return True
            else:
                print(f"Param {name} set attempt {attempt} not confirmed (got={got})")
        except Exception as e:
            print(f"Warning: could not read param {name} after set attempt {attempt}: {e}")
    return False

def wait_for_armable(vehicle, timeout=ARMABLE_WAIT):
    print(f"Waiting up to {timeout}s for vehicle.is_armable (EKF & sensors)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            if vehicle.is_armable:
                print("Vehicle is armable.")
                return True
        except Exception:
            # vehicle.is_armable can raise if vehicle not fully initialised
            pass
        # print minimal status for user
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.5)
    print("\nTimed out waiting for vehicle.is_armable.")
    return False

def diagnostic_print(vehicle):
    try:
        print("vehicle.is_armable:", vehicle.is_armable)
    except Exception:
        pass
    try:
        print("vehicle.mode:", vehicle.mode.name)
    except Exception:
        pass
    try:
        print("vehicle.armed:", vehicle.armed)
    except Exception:
        pass
    try:
        print("battery:", vehicle.battery)
    except Exception:
        pass
    # Print recent messages if accessible (best-effort)
    try:
        # DroneKit doesn't expose autopilot messages easily; the console will show them.
        pass
    except Exception:
        pass

def main():
    if not confirm_user():
        print("User did not confirm. Exiting.")
        return

    conn = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONN
    baud = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_BAUD

    print("Connecting to", conn, "baud", baud)
    vehicle = None
    try:
        vehicle = connect(conn, baud=baud, wait_ready=True, heartbeat_timeout=30, timeout=30)
        print("Connected: mode:", vehicle.mode.name, "armed:", vehicle.armed)
        print("Battery:", vehicle.battery)

        # Save original parameters to restore later (best-effort)
        orig = {}
        for p in ('ARMING_CHECK', 'BRD_SAFETYENABLE', 'FS_THR_ENABLE', 'FS_GCS_ENABLE'):
            try:
                orig[p] = vehicle.parameters.get(p)
            except Exception:
                orig[p] = None
        print("Saved params:", orig)

        # Quick battery check (safe bench threshold - adjust for your battery)
        vb = None
        try:
            vb = vehicle.battery.voltage
        except Exception:
            pass
        if vb is not None:
            print("Battery voltage:", vb)
            if vb < 7.0:
                print("Battery voltage too low for bench test. Aborting.")
                return

        # Wait until EKF/sensors ready (vehicle.is_armable)
        if not wait_for_armable(vehicle, timeout=ARMABLE_WAIT):
            print("Diagnostics:")
            diagnostic_print(vehicle)
            print("Aborting because vehicle is not armable. Check EKF/IMU/calibration and ensure vehicle is level and stationary on boot.")
            return

        # Set bench-safe params with retries (disable arming checks and radio failsafe)
        print("Setting bench-safe parameters (bench only). Will restore at the end.")
        bench_changes = {
            'ARMING_CHECK': 0,
            'BRD_SAFETYENABLE': 0,
            'FS_THR_ENABLE': 0,
            'FS_GCS_ENABLE': 0
        }
        for name, val in bench_changes.items():
            success = try_set_param(vehicle, name, val)
            if not success:
                # Not critical for all params but warn user
                print(f"Warning: failed to confirm setting {name} -> {val}. You may need to set it manually via Mission Planner/MAVProxy.")
        # allow autopilot to process the new params
        time.sleep(1.0)

        # Ensure a safe mode for bench
        try:
            vehicle.mode = VehicleMode("STABILIZE")
            time.sleep(0.5)
        except Exception:
            pass

        # Arm the vehicle (after is_armable)
        print("Arming vehicle...")
        vehicle.armed = True
        t0 = time.time()
        while not vehicle.armed and (time.time() - t0) < 12:
            print("Waiting for arm...")
            time.sleep(0.5)
        if not vehicle.armed:
            print("Failed to arm. Aborting and will restore parameters.")
            diagnostic_print(vehicle)
            return
        print("ARMED!")

        # Ramp throttle up gradually
        print(f"Ramping throttle up to {TEST_THROTTLE}. Step {RAMP_STEP}, delay {RAMP_DELAY}s.")
        pwm = 1000
        vehicle.channels.overrides['3'] = pwm
        time.sleep(0.2)
        while pwm < TEST_THROTTLE:
            pwm = min(TEST_THROTTLE, pwm + RAMP_STEP)
            vehicle.channels.overrides['3'] = int(pwm)
            print("Throttle ->", pwm)
            time.sleep(RAMP_DELAY)

        print("Holding at target for", HOLD_SECONDS, "s...")
        for i in range(HOLD_SECONDS):
            print("Hold", i+1)
            time.sleep(1)

        # Ramp down
        print("Ramping down to idle...")
        while True:
            cur = vehicle.channels.overrides.get('3') or pwm
            cur = int(cur)
            if cur <= 1000:
                break
            cur = max(1000, cur - RAMP_STEP)
            vehicle.channels.overrides['3'] = cur
            print("Throttle ->", cur)
            time.sleep(0.25)

        vehicle.channels.overrides = {}
        print("Cleared overrides. Disarming...")
        vehicle.armed = False
        t0 = time.time()
        while vehicle.armed and (time.time() - t0) < 8:
            time.sleep(0.2)
        print("Disarmed.")

    except KeyboardInterrupt:
        print("\nUser interrupted. Attempting to safely clean up.")
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
    finally:
        # Best-effort restore of original params
        if vehicle is not None:
            print("Restoring original parameters (best-effort)...")
            for p, v in orig.items():
                try:
                    if v is None:
                        continue
                    vehicle.parameters[p] = v
                    print(f"Restored {p} -> {v}")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Warning: could not restore {p}: {e}")
            # clear overrides and close
            try:
                vehicle.channels.overrides = {}
            except Exception:
                pass
            try:
                vehicle.close()
            except Exception:
                pass
            print("Vehicle connection closed.")
        print("Done. IMPORTANT: re-enable safety/arming checks before flight (ARMING_CHECK and BRD_SAFETYENABLE).")

if __name__ == "__main__":
    main()