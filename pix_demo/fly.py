from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import socket
import argparse


def connectMyCopter():
        parser = argparse.ArgumentParser(description='commands')
        parser.add_argument('--connect')
        args = parser.parse_args()

        conns = args.connect
        baud_rate = 57600

        veh = connect(conns,baud=baud_rate,wait_ready=True)
        return veh


def arm():
        while veh.is_armable==False:
                print("Waiting to arm..")
                time.sleep(1)
        print("Veh is armable")
        veh.armed=True
        while veh.armed == False:
                print("Waiting to arm...")
                time.sleep(1)
        print("Veh is now armed")
        print("CMG props are spinning")

        return None

veh = connectMyCopter()
arm()
print("EOS")