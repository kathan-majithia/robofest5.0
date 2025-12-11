from dronekit import connect
import sys, time

conn = sys.argv[1] if len(sys.argv)>1 else "/dev/ttyACM0"
baud = int(sys.argv[2]) if len(sys.argv)>2 else 57600

print("Connecting to", conn, "baud", baud)
v = connect(conn, baud=baud, wait_ready=True, timeout=20)

print("Vehicle mode:", v.mode.name, "Armed:", v.armed)
print("Battery:", v.battery)
print("GPS:", v.gps_0)
print("Parameters downloaded:", len(v.parameters))

# Print RC channels (mapping) and raw values if available
print("Channels (mapping):", v.channels)      # dict-like: '1','2',...
try:
    raw = v.channels.raw
    print("Channels (raw):", raw)            # may show numerical PWM values
except Exception:
    pass

# Loop to watch updates for 10 seconds
print("\nWatching RC channels for 10s (press Ctrl-C to stop):")
end = time.time()+10
while time.time()<end:
    print(time.time(), "->", v.channels)
    time.sleep(0.5)

v.close()
print("Done")