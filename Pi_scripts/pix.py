from dronekit import connect,VehicleMode
import time

print("Connecting...")
veh = connect('/dev/ttyACM0',baud=115200,wait_ready=True)

print("Connected to Pixhawk!")

print("Mode : ",veh.mode.name)

print("Armed : ",veh.armed)
