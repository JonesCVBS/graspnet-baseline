import cv2
import numpy as np

#This file will take images and joint positions of a ur3e robot and perform handeye calibration
#It must take the images, find the chessboard and with that perform intrinsic and extrinsic calibration
#form the joint positions it will calculate the transformation from the base to the end effector
#It will then use the open cv library to calculate the transformation between the camera and the end effector

def loadCameraImgs(Img_path):
    """Loads all of the images in order (they're ordered like RGBimg_001,...RGBImg_0013,....) for camera calibration,
    the ammount of images is variable but they're all the same size"""
    #Load the images
    img_array = []
    #find number of images and save in a variable named TotalImages by reading the directory for files ending in .png
    TotalImages = len([name for name in os.listdir(Img_path) if name.endswith(".png")])
    for i in range(1,TotalImages):
        filename = Img_path + "RGBImg_" + str(i).zfill(3) + ".png"
        img = cv2.imread(filename)
        img_array.append(img)
    return img_array

def loadJointPos(JointPos_path):
    """Loads the joint position data from the .npz files and returns a list of joint positions"""
    #Load the joint positions
    JointPos_array = []
    #find number of images and save in a variable named TotalImages by reading the directory for files ending in .npz
    TotalImages = len([name for name in os.listdir(JointPos_path) if name.endswith(".npz")])
    for i in range(1,TotalImages):
        filename = JointPos_path + "JointPos_" + str(i).zfill(3) + ".npz"
        JointPos = np.load(filename)
        JointPos_array.append(JointPos)
    return JointPos_array
