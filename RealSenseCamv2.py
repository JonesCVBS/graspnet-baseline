import pyrealsense2 as rs
import numpy as np
import open3d as o3d

class RealSenseCamera:
    def __init__(self):
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()

        # Create a config object and configure the pipeline to stream
        # both color and depth frames at 640x480 resolution
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)

        # Get the intrinsics of the color camera
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # Create an align object
        self.align = rs.align(rs.stream.color)

    def __del__(self):
        # Stop the pipeline on object deletion
        self.pipeline.stop()

    def get_point_cloud(self):
        # Capture a frame
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = self.align.process(frames)

        # Get the aligned depth frame and color frame
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        #convert from RGB to BGR
        color_image = np.asanyarray(color_frame.get_data())
        color_image = color_image[:, :, ::-1]

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create a point cloud from the depth and color images
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())
        num_points = depth_image.shape[0] * depth_image.shape[1]
        #print num_points
        #print("vtx.shape: ", vtx.shape)
        #print("num_points: ", num_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx.tolist())
        pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        return pcd, color_image, depth_image

    def save_point_cloud(self, filename):
        # Get the point cloud and save it to disk
        pcd, _, _ = self.get_point_cloud()
        o3d.io.write_point_cloud(filename, pcd)

    def save_color_image(self, filename):
        # Capture a frame and save the color image to disk
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        o3d.io.write_image(filename, o3d.geometry.Image(color_image))

    def save_depth_image(self, filename):
        # Capture a frame and save the depth image to disk
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        o3d.io.write_image(filename, o3d.geometry.Image(depth_image))

    def VisualizePtcWithVector(self, pcd,ShowAxis=False):
        """Visualize point cloud"""
        o3d.visualization.draw_geometries([pcd])
        #draw axis
        if ShowAxis:
            o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])])

    def VisualizePtcWithVectorAndGrasp(self, pcd, grasp=None):
        """Visualize point cloud and, if defined, shows the vector in the point cloud"""

        def euler_to_direction_vector(angles):
            """Converts Euler angles (roll, pitch, yaw) to a unit direction vector."""
            roll, pitch, yaw = angles
            x = np.cos(yaw) * np.cos(pitch)
            y = np.sin(yaw) * np.cos(pitch)
            z = np.sin(pitch)
            return np.array([x, y, z])

        if grasp is not None:
            # Draw the point cloud, the grasp, and the axis
            # The grasp is a 6D vector; the first 3 are the Euler angles (rx, ry, rz),
            # and the last three are the position (x, y, z)

            # Create an Open3D LineSet for the grasp vector
            grasp_line = o3d.geometry.LineSet()

            # Convert Euler angles to a unit direction vector
            direction = euler_to_direction_vector(grasp[:3])

            # Calculate the end point of the grasp vector
            scale = 0.1  # You can adjust this value to change the length of the vector
            start_point = np.array(grasp[3:6])
            end_point = start_point + scale * direction

            # Set the points and lines of the LineSet
            grasp_line.points = o3d.utility.Vector3dVector([start_point, end_point])
            grasp_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            grasp_line.colors = o3d.utility.Vector3dVector(
                [[1, 0, 0]])  # Set the color of the vector (red in this case)

            o3d.visualization.draw_geometries(
                [pcd, grasp_line, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])])
        else:
            # Draw the point cloud and the axis
            o3d.visualization.draw_geometries(
                [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])])
#
if __name__=='__main__':
    cam = RealSenseCamera()
    #get point cloud
    pcd, color_image, depth_image = cam.get_point_cloud()
    #show point cloud
    o3d.visualization.draw_geometries([pcd])
    #save point cloud
    cam.save_point_cloud('point_cloud.ply')
    cam.save_color_image('color_image.png')
    cam.save_depth_image('depth_image.png')
    #Get the intrinsic parameters of the color camera
    print("Intrinsics of the color camera: ", cam.intrinsics)