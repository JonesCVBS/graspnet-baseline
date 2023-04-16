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
import pyrealsense2 as rs


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data():
    cam = RealSenseCamera()
    # get point cloud
    #Take 10 point clouds using cam.get_point_cloudd() and average them out
    #point_cloud, color_image, depth_image = cam.get_point_cloud()
    #point_cloud is an open3d point cloud object
    #color_image is a numpy array of shape (480, 640, 3)
    #depth_image is a numpy array of shape (480, 640)
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

    #sample points from point cloud
    Numpy_point_cloud = np.array(point_cloud.points)
        # sample points
    if len(Numpy_point_cloud) >= cfgs.num_point:
        idxs = np.random.choice(len(Numpy_point_cloud), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(Numpy_point_cloud))
        idxs2 = np.random.choice(len(Numpy_point_cloud), cfgs.num_point - len(Numpy_point_cloud), replace=True)

    #Get end_points convert them to torch tensors
    #point_cloud needs to be converted and renamed.
    Numpy_point_cloud = Numpy_point_cloud[idxs]
    color_image = color_images/10
    color_image

    end_points = {}
    end_points['point_clouds'] = torch.from_numpy(Numpy_point_cloud).unsqueeze(0).float().cuda()
    end_points['cloud_colors'] = torch.from_numpy(color_image).unsqueeze(0).float().cuda()

    return end_points, point_cloud


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    print('Showing top 50 grasps')
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def get_grasp_vectors(gg):
    grasp_group_array = gg.grasp_group_array
    grasp_vectors = np.zeros((len(grasp_group_array), 6))
    for i in range(len(grasp_group_array)):
        rotation_matrix = gg.rotation_matrices[i]
        rotation_vector = cv2.Rodrigues(rotation_matrix)[0].flatten()
        translation_vector = gg.translations[i]
        grasp_vectors[i, :] = np.concatenate((rotation_vector, translation_vector))
    grasp_scores = gg.scores
    return grasp_vectors, grasp_scores

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data()
    gg = get_grasps(net, end_points)
    #print(gg)
    print(dir(gg))
    grasp_vectors, grasp_scores = get_grasp_vectors(gg)
    print('Grasp vectors:', grasp_vectors)
    print('Grasp scores:', grasp_scores)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)
    print(gg.object_ids)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
