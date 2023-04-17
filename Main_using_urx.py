import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'graspnet-baseline'))
from GraspPointsFromRealsense import GraspNetDemo
from RealSenseCamv2 import RealSenseCamera
import numpy as np
import socket
import pickle
import cv2


"""This is the main file for testing the grasping application using a Ur3e robot and a Realsense camera."""

#Init the graspnet class
graspnet = GraspNetDemo(checkpoint_path='graspnet-baseline/checkpoint-rs.tar')
T_Cam2EE = np.load('HandeyeCalibration/T_cam2gripper_Method_2.npz')['arr_0']
#TEE2Cam = T_Cam2EE
TEE2Cam = np.load('HandeyeCalibration/T_gripper2cam_Method_2.npz')['arr_0']

print("TEE2Cam is: ", TEE2Cam)
print("T_Cam2EE is: ", T_Cam2EE)

import urx
import numpy as np

rob = urx.Robot("192.168.1.10")
rob.set_tcp((0.03, 0, 0.1739,0, 0, 0, ))
rob.set_tcp((0.0091, -0.004, 0.1439,0, 0, 0, ))
rob.set_tcp((0, -0, 0,0, 0, 0, ))

rob.set_payload(1.2, (0, -0.0013, 0.055))

#Move to initial position
Pose_table = np.array([-0.42292605641053516, -0.043388532280434156, 0.3354523516505133, 2.361213727001138, 2.072139846990468, 4.295346272191453e-05])
Pose_table_angled = np.array([0.15170553020713323, -0.12156380800122886, 0.35432814878277663, 0.34710206111963327, -3.0699030615481946, 0.3695983305307578])
Pose_robot_ws = np.array([0.21844737778746745, -0.21545469618691046, 0.2925104933944713, -1.190170844575031, 2.9074123639602196, 1.58705773960657e-05])
initalPos = Pose_table
# move the robot
a = 0.1
v = 0.1
rob.movel((initalPos), a, v)  # move relative to current pose
T_gripper2grasp = np.eye(4)

def create_test_grasping_points():
    """Uses graspnet to create a grasping point"""
    R_trans, poses, grasp_scores = graspnet.demo()   #obtain the grasp vectors and scores
    # extracting top grasp
    if len(grasp_scores) == 0 or grasp_scores[np.argmax(grasp_scores)]<0.7:
        print(grasp_scores)
        T_gripper2grasp = None
        print("No grasp found")
        return

    #All poses picked are good, usually arround here you would have some sort of inverse kinematic solver, in here we're
    # just gonna chose the pose most vertical, as that one is usually the easier one

    best_grasp_index = graspnet.PosePicker(R_trans,poses,grasp_scores)
    best_grasp_orientation = R_trans[best_grasp_index]  # Rotation matrix (3x3)
    best_grasp_position = poses[best_grasp_index]  # Translation (x, y, z) coordinates
    print("Best grasp orientation: ", best_grasp_orientation)
    print("Best grasp position: ", best_grasp_position)
    #input("TESTING. Press Enter to continue...")
    #converting orientation of the projected gripepr to correct frame
    #R_object2cam turns it so that the frame result is aligned with the camera frame
    #R_cam2ur so that the camera frame is aligned with the ur gripper pose
    R_object2cam = np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]])
    R_cam2ur = np.array([[-1,  0,  0],
                         [ 0, -1,  0],
                         [ 0,  0,  1]])
    R_cam2ur = np.eye(3)
    R_correction = np.matmul(R_object2cam,R_cam2ur)
    best_grasp_orientation = np.matmul( best_grasp_orientation,R_correction)
    #The current grasping point is in the camera frame. It must be converted to the robot base frame
    T_cam2grasp = np.eye(4)
    T_cam2grasp[:3,:3] = best_grasp_orientation
    T_cam2grasp[:3,3] = np.transpose(best_grasp_position)

    # to get the transform from the EE to cam we need to multiply TEE2Cam*TCam2Grasp
    T_EE2grasp = np.matmul(TEE2Cam, T_cam2grasp)
    print("T_EE2grasp is: ", T_EE2grasp)
    #add the offset to the grasping point, no difference in orientation
    T_Gripper2EE = np.eye(4)
    #T_Gripper2EE[0,3] = 0.05
    T_Gripper2EE[2,3] = -0.1239
    T_gripper2grasp = np.matmul(T_Gripper2EE,T_EE2grasp)
    print("T_gripper2grasp is:", T_gripper2grasp)
    return T_gripper2grasp

def axis_angle_to_rotation_matrix(rv):
    angle = np.linalg.norm(rv)
    if angle < 1e-6:
        return np.identity(3)

    axis = rv / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def rotation_matrix_to_axis_angle(R):
        """Converts a rotation matrix to an axis angle representation for a ur robot"""
        print("R is ", R)
        axis = np.zeros(3)
        angle = np.arccos((np.trace(R) - 1) / 2)

        axis[0] = R[2, 1] - R[1, 2]
        axis[1] = R[0, 2] - R[2, 0]
        axis[2] = R[1, 0] - R[0, 1]

        norm = np.linalg.norm(axis)

        if norm > 1e-6:
            axis /= norm
        else:
            axis = np.array([1, 0, 0])

        rotation_vector = axis * angle
        print("rotation_vector is ", rotation_vector)
        return rotation_vector

Items = 1
while Items==1:
    #This creates the grasp point
    T_gripper2grasp = create_test_grasping_points()

    if T_gripper2grasp is None:
        Items = 0
        print("No grasp found")
        rob.close()
        sys.exit()

    #Now that grasp point is converted into the correct reference frame
    #Show T_gripper2grasp as a vector
    R_gripper2Grasp = T_gripper2grasp[:3,:3]
    Rvec_gripper2Grasp = rotation_matrix_to_axis_angle(R_gripper2Grasp)
    p_gripper2Grasp = T_gripper2grasp[:3,3]
    print("the gripper2grasp vector is: ", np.concatenate((p_gripper2Grasp, Rvec_gripper2Grasp)))
    print("Current tool pose is: ",  rob.getl())

    a = 0.1
    v = 0.1
    CurrentPose = rob.getl()
    #Convert the current pose into a 4x4 transform

    Rvec_current = CurrentPose[3:]
    R_current = axis_angle_to_rotation_matrix(Rvec_current)
    print("R_current is:", R_current)
    p_current = CurrentPose[:3]
    T_current = np.eye(4)
    T_current[:3,:3] = R_current
    T_current[:3,3] = p_current

    #multiplying the matrices to get the goal pose
    T_goal = np.matmul(T_current, T_gripper2grasp)
    p_goal = T_goal[:3,3]
    R_goal = T_goal[:3,:3]

    #converting the goal pose into a ur robot vector
    Rvec_goal = rotation_matrix_to_axis_angle(R_goal)
    goalPoint = np.zeros(6)
    goalPoint[:3] = np.transpose(p_goal)
    goalPoint[3:] = Rvec_goal
    print("The goal position is:" , goalPoint)

    # move the robot to grasp a bit higher than the point
    goalPoint_High = np.array(goalPoint)
    goalPoint_High[2] += 0.05
    rob.movel((goalPoint_High), a, v)  # move relative to current pose
    rob.movel((goalPoint), a, v)  # move relative to current pose

    #wait for input from user
    input("CLOSE GRIPPER. Press Enter to continue...")

    #move back to starting position
    rob.movel((initalPos), a, v)  # move relative to current pose
    Pose_drop = np.array([0.36468467427686435, 0.015347861217733641, 0.2050860898047682, -1.8067321353604489, 2.527139861231424, -0.09850408959046032])
    Pose_grop_table = np.array([-0.09081074220933416, -0.32326444339291244, 0.25236927286485307, -2.2726437002844895, -2.100567158139399, -0.09742738086098596])
    rob.movel((Pose_grop_table), a, v)  # move relative to current pose
    # wait for input from user
    input("CLOSE GRIPPER. Press Enter to continue...")
    rob.movel((initalPos), a, v)  # move relative to current pose