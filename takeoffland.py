#FOR CAPTURTING PICTURES

from djitellopy import Tello
import cv2
import time
global img

tello = Tello()
tello.connect()
tello.takeoff()


    

tello.end()