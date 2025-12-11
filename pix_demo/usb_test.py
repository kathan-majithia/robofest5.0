from dronekit import connect, VehicleMode
import time

veh = connect('/dev/ttyACM0',baud=115200,wait_ready=True,timeout=60)
print("Connected mode : ",veh.mode.name," Armed : ",veh.armed)

print("Paramaters downloaded : ",len(veh.parameters))

veh.close()