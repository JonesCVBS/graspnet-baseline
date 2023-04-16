""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from RealSenseCamv2 import RealSenseCamera



class GraspNetDemo:
    def __init__(self, checkpoint_path, num_point=20000, num_view=300, collision_thresh=0.9, voxel_size=0.003):
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self):
        cam = RealSenseCamera()
        # get point cloud
        # Take 10 point clouds using cam.get_point_cloudd() and average them out
        # point_cloud, color_image, depth_image = cam.get_point_cloud()
        # point_cloud is an open3d point cloud object
        # color_image is a numpy array of shape (480, 640, 3)
        # depth_image is a numpy array of shape (480, 640)
        for i in range(10):
            point_cloud, color_image, depth_image = cam.get_point_cloud()
            if i == 0:
                point_clouds = point_cloud
                color_images = color_image
                depth_images = depth_image
            else:
                point_clouds += point_cloud
                color_images += color_image
                depth_images += depth_image

        # sample points from point cloud
        Numpy_point_cloud = np.array(point_cloud.points)
        # sample points
        if len(Numpy_point_cloud) >= self.num_point:
            idxs = np.random.choice(len(Numpy_point_cloud), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(Numpy_point_cloud))
            idxs2 = np.random.choice(len(Numpy_point_cloud), self.num_point - len(Numpy_point_cloud), replace=True)

        # Get end_points convert them to torch tensors
        # point_cloud needs to be converted and renamed.
        Numpy_point_cloud = Numpy_point_cloud[idxs]
        color_image = color_images / 10
        color_image

        end_points = {}
        end_points['point_clouds'] = torch.from_numpy(Numpy_point_cloud).unsqueeze(0).float().cuda()
        end_points['cloud_colors'] = torch.from_numpy(color_image).unsqueeze(0).float().cuda()

        return end_points, point_cloud

    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self,gg, cloud,num_grasps):
        print('Showing top 50 grasps')
        gg.nms()
        gg.sort_by_score()
        gg = gg[:num_grasps]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def get_grasp_vectors(self,gg):
        grasp_group_array = gg.grasp_group_array
        grasp_vectors = np.zeros((len(grasp_group_array), 6))
        R_trans = gg.rotation_matrices
        poses = gg.translations
        grasp_scores = gg.scores
        return R_trans, poses, grasp_scores

    def demo(self):
        net = self.get_net()
        end_points, cloud = self.get_and_process_data()
        gg = self.get_grasps(net, end_points)
        # print(gg)
        print(dir(gg))
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        self.vis_grasps(gg, cloud,30)
        R_trans, poses, grasp_scores = self.get_grasp_vectors(gg)

        #Remove poses with widths below a threshold
        maxwidth = 0.09
        R_trans = R_trans[gg.widths < maxwidth]
        poses = poses[gg.widths < maxwidth]
        grasp_scores = grasp_scores[gg.widths < maxwidth]
        #Pickbest and show
        best_grasp_index = self.PosePicker(R_trans,poses, grasp_scores)
        Best_graspscore = grasp_scores[best_grasp_index]
        gg.scores[gg.scores==Best_graspscore] += 50
        self.vis_grasps(gg, cloud,1)
        print(np.array(gg.rotation_matrices).shape)
        print(np.array(gg.translations).shape)
        print("The grasp scores are:", gg.scores)
        print("Grasp group array shape is ", gg.grasp_group_array.shape)
        return R_trans, poses, grasp_scores


    def PosePicker(self,R_trans,t_trans,grasp_scores):
        """Calculates the angles of rotation between all the matrices, the one with th smallest is the output
        This is done to find the pose that is most vertical"""

        #Check if there are any grasp poses:
        if len(R_trans)==0:
            return

        #initiate angles list
        Angles = []
        Dist = []
        Score = []
        #the correction so the rotation in the camera frame matches the camera frame
        R_correction = np.array([[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])
        for i in range(0,len(R_trans)):

            t_i = t_trans[i]                        #get the translation

            R_i = np.matmul(R_trans[i],R_correction)    #so the frame is correct
            score_i = grasp_scores[i]                #get the score of the grasp
            if score_i > 0.7:
                # calculate the distance of the translation
                Dist_i = np.sqrt(t_i[0] ** 2 + t_i[1] ** 2 + t_i[2] ** 2)
                #Calculate the angle of rotation
                Angle_i = np.arccos((np.trace(R_i) - 1) / 2)
                #
                Angles.append(Angle_i)
                Dist.append(Dist_i)
            else:
                Angles.append(-1)
                Dist.append(-1)


        #Normalize the angles and distances
        Angles = np.array(Angles)
        Dist = np.array(Dist)
        Angles = Angles/np.max(Angles)
        Dist = Dist/np.max(Dist)
        #change the values with -1 into 1
        Dist[Dist == -1] = 1
        Angles[Angles == -1] = 1
        #Add the angles and distances
        Scores = 2*Angles + Dist  #the smaller the better
        #return the index of the smallest angle
        print("angles are: ",Angles)
        print("Dists are:", Dist)
        print("Scores are:",Scores)
        BestScore_index = np.argmin(Scores)
        print("BestScore_index is:", BestScore_index)
        BestScore = Scores[BestScore_index]
        print("BestScore is:", BestScore)


        return BestScore_index

if __name__=='__main__':
    cfgs = {
        'checkpoint_path': 'path/to/checkpoint',
        'num_point': 20000,
        'num_view': 300,
        'collision_thresh': 0.8,
        'voxel_size': 0.01
    }
    data_dir = 'doc/example_data'
    grasp_net_demo = GraspNetDemo(checkpoint_path="checkpoint-rs.tar")
    data_dir = 'doc/example_data'
    R_trans, poses, grasp_scores = grasp_net_demo.demo()



    #extracting top grasp
    best_grasp_index = grasp_net_demo.PosePicker(R_trans,poses,grasp_scores)
    best_grasp_orientation = R_trans[best_grasp_index]  # Rotation matrix (3x3)
    best_grasp_position = poses[best_grasp_index]  # Translation (x, y, z) coordinates
    print("Best grasp orientation: ", best_grasp_orientation)
    print("Best grasp position: ", best_grasp_position)

    #this next part is for testing
    #get the point cloud
    cam = RealSenseCamera()
    point_cloud, color_image, depth_image = cam.get_point_cloud()

    #plotting the reference frame defined by best_grasp_orientation at the location of best_grasp_position in the point cloud
    #best_grasp_orientation is a 3x3 rotation matrix that needs to be converted into 3 vectors to plot
    #best_grasp_position is a 3x1 vector that contains the positon x,y,z of the grasp

    #with te correction
    # adding a rotation matrix
    R_correction = np.array([[ 0, 0,  1],
              [ 1, 0, 0],
              [ 0, 1,  0]])
    #R_correction = np.eye(3)
    #R_correction = np.matmul(R_correction_xaxis, R_correction_yaxis)
    print("R_correction: ", R_correction)

    #multiplying matrices using
    R_result = np.matmul( best_grasp_orientation,R_correction)
    print("R_result: ", R_result)
    # Create a coordinate frame with the given rotation and position
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    frame.rotate(best_grasp_orientation, center=frame.get_center())
    frame.translate(best_grasp_position, relative=False)

    #Create 2nd frame
    # Create a coordinate frame with the given rotation and position
    frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    frame2.rotate(R_result, center=frame.get_center())
    frame2.translate(best_grasp_position, relative=False)

    #point cloud frame
    frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Create a visualizer and add the point cloud and the reference frame
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(frame)
    vis.add_geometry(frame2)
    vis.add_geometry(frame3)

    # Run the visualizer
    vis.run()

    # Clean up
    vis.destroy_window()


