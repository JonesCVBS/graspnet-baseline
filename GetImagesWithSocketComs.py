import socket
import pickle
import random
import sys
from RealSenseCamv2 import RealSenseCamera
import cv2


def SaveRGBImage(i,ShowImage=False):
    cam = RealSenseCamera()
    # get point cloud
    pcd, color_image, depth_image = cam.get_point_cloud()
    #save color image where i is the image number, with 3 digits
    cam.save_color_image("color_image" + str(i).zfill(3) + ".png")
    if ShowImage:
        #show color_image using opencv
        cv2.imshow('color_image',color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return i

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)
print("Waiting for connection...")
i = 1
try:
    while True:
        connection, client_address = server_socket.accept()
        try:
            print("Connection from", client_address)
            data = connection.recv(1024)
            if data:
                i = SaveRGBImage(i,ShowImage=True)
                serialized_data = pickle.dumps(i, protocol=2)
                connection.sendall(serialized_data)
                i = i+1
        finally:
            connection.close()
except KeyboardInterrupt:
    print("\nShutting down server...")
    server_socket.close()
    sys.exit(0)
