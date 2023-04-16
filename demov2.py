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



class GraspNetDemo:
    def __init__(self, checkpoint_path, num_point=20000, num_view=300, collision_thresh=0.01, voxel_size=0.01):
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

    def get_and_process_data(self,data_dir):
        # load data
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        print("cloud_masked type: ", cloud_masked.dtype)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

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

    def vis_grasps(self,gg, cloud):
        print('Showing top 50 grasps')
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def get_grasp_vectors(self,gg):
        grasp_group_array = gg.grasp_group_array
        grasp_vectors = np.zeros((len(grasp_group_array), 6))
        for i in range(len(grasp_group_array)):
            rotation_matrix = gg.rotation_matrices[i]
            rotation_vector = cv2.Rodrigues(rotation_matrix)[0].flatten()
            translation_vector = gg.translations[i]
            grasp_vectors[i, :] = np.concatenate((rotation_vector, translation_vector))
        grasp_scores = gg.scores
        return grasp_vectors, grasp_scores

    def demo(self,data_dir):
        net = self.get_net()
        end_points, cloud = self.get_and_process_data(data_dir)
        gg = self.get_grasps(net, end_points)
        # print(gg)
        print(dir(gg))


        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))

        grasp_vectors, grasp_scores = self.get_grasp_vectors(gg)
        self.vis_grasps(gg, cloud)
        print(gg.object_ids)
        return grasp_vectors, grasp_scores


if __name__=='__main__':
    cfgs = {
        'checkpoint_path': 'path/to/checkpoint',
        'num_point': 20000,
        'num_view': 300,
        'collision_thresh': 0.01,
        'voxel_size': 0.01
    }
    data_dir = 'doc/example_data'
    grasp_net_demo = GraspNetDemo(checkpoint_path="checkpoint-rs.tar")
    data_dir = 'doc/example_data'
    grasp_vectors, grasp_scores = grasp_net_demo.demo(data_dir)
    print('Grasp vectors:', grasp_vectors)
    print('Grasp scores:', grasp_scores)
