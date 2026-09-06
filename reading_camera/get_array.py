import cv2
from picamera2 import Picamera2


def init():
    camera= Picamera2()
    config= camera.create_preview_configuration(main={"format":"", "size":(214,214)})
    camera.configure(config)
    camera.start()

def get_frame():
    frame= camera.capture_array()
    return frame