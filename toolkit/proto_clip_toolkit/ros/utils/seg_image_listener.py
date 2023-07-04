#!/usr/bin/env python
"""ROS image listener"""

import os, sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
from scipy.io import savemat

import rospy
import tf
import tf2_ros
import message_filters
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from .ros_utils import ros_qt_to_rt

from .segmentation_utils import visualize_segmentation

lock = threading.Lock()

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self, camera='Fetch'):

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        # initialize a node
        self.tf_listener = tf.TransformListener()        

        if camera == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame        
        elif camera == 'Realsense':
            # use RealSense camera
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            print('camera %s is not supported in image listener' % camera)
            sys.exit(1)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):
    
        # get camera pose in base
        try:
             trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
             RT_camera = ros_qt_to_rt(rot, trans)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None             

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera


    def get_data(self):

        with lock:
            if self.im is None:
                return None, None, None, None, None, self.intrinsics
            im_color = self.im.copy()
            depth_image = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera.copy()

        xyz_image = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width)
        xyz_array = xyz_image.reshape((-1, 3))
        xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
        xyz_base = xyz_base.T.reshape((self.height, self.width, 3))
        return im_color, depth_image, xyz_image, xyz_base, RT_camera, self.intrinsics


# class to recieve images and segmentation labels
class SegImageListener:

    def __init__(self, data_dir):

        self.im = None
        self.depth = None
        self.depth_frame_id = None
        self.depth_frame_stamp = None
        self.xyz_image = None
        self.label = None
        self.bbox = None
        self.score = None
        self.counter = 0
        self.cv_bridge = CvBridge()
        self.base_frame = 'base_link'
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)        
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        label_sub = message_filters.Subscriber('/seg_label_refined', Image, queue_size=10)
        score_sub = message_filters.Subscriber('/seg_score', Image, queue_size=10)          
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame        

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print(intrinsics)
        
        # camera pose in base
        transform = self.tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time(0),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        # print(self.camera_pose)

        queue_size = 1
        slop_seconds = 3.0
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, label_sub, score_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        
        # data saving directory
        now = datetime.datetime.now()
        seq_name = "listener_{:%m%dT%H%M%S}/".format(now)
        self.save_dir = os.path.join(data_dir, seq_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        


    def callback_rgbd(self, rgb, depth, label, score):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        label = self.cv_bridge.imgmsg_to_cv2(label)
        score = self.cv_bridge.imgmsg_to_cv2(score)
        
        # compute xyz image
        height = depth_cv.shape[0]
        width = depth_cv.shape[1]
        xyz_image = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)
        
        # compute the 3D bounding box of each object
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = len(mask_ids)
        # print('%d objects segmented' % num)
        bbox = np.zeros((num, 8), dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)          

        for index, mask_id in enumerate(mask_ids):
            mask = np.array(label == mask_id).astype(np.uint8)
            
            # erode mask
            mask2 = cv2.erode(mask, kernel)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 2, 1)
            # plt.imshow(mask)
            # ax = fig.add_subplot(1, 2, 2)
            # plt.imshow(mask2)
            # plt.show()               
            
            mask = (mask2 > 0) & (depth_cv > 0)
            points = xyz_image[mask, :]
            confidence = np.mean(score[mask])
            # convert points to robot base
            points_base = np.matmul(self.camera_pose[:3, :3], points.T) + self.camera_pose[:3, 3].reshape((3, 1))
            points_base = points_base.T
            center = np.mean(points_base, axis=0)
            if points_base.shape[0] > 0:
                x = np.max(points_base[:, 0]) - np.min(points_base[:, 0])
                y = np.max(points_base[:, 1]) - np.min(points_base[:, 1])
                # deal with noises in z values
                z = np.sort(points_base[:, 2])
                num = len(z)
                percent = 0.05
                lower = int(num * percent)
                upper = int(num * (1 - percent))
                if upper > lower:
                    z_selected = z[lower:upper]
                else:
                    z_selected = z
                z = np.max(z_selected) - np.min(z_selected)
            else:
                x = 0
                y = 0
                z = 0
            bbox[index, :3] = center
            bbox[index, 3] = x
            bbox[index, 4] = y
            bbox[index, 5] = z
            bbox[index, 6] = confidence
            bbox[index, 7] = mask_id
            
        # filter box
        index = bbox[:, 5] > 0
        bbox = bbox[index, :]

        with lock:
            self.im = im.copy()        
            self.label = label.copy()
            self.score = score.copy()
            self.depth = depth_cv.copy()
            self.depth_frame_id = depth.header.frame_id
            self.depth_frame_stamp = depth.header.stamp
            self.xyz_image = xyz_image
            self.bbox = bbox            

            
    # save data
    def save_data(self, step: int):
        # save meta data
        factor_depth = 1000.0        
        meta = {'intrinsic_matrix': self.intrinsics, 'factor_depth': factor_depth, 'camera_pose': self.camera_pose}
        filename = self.save_dir + 'meta-{:06}.mat'.format(step)
        savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))

        # convert depth to unit16
        depth_save = np.array(self.depth * factor_depth, dtype=np.uint16)

        # segmentation label image
        im_label = visualize_segmentation(self.im, self.label, return_rgb=True)

        save_name_rgb = self.save_dir + 'color-{:06}.jpg'.format(step)
        save_name_depth = self.save_dir + 'depth-{:06}.png'.format(step)
        save_name_label = self.save_dir + 'label-{:06}.png'.format(step)
        save_name_label_image = self.save_dir + 'gt-{:06}.jpg'.format(step)        
        save_name_score = self.save_dir + 'score-{:06}.png'.format(step)        
        cv2.imwrite(save_name_rgb, self.im)
        cv2.imwrite(save_name_depth, depth_save)
        cv2.imwrite(save_name_label, self.label.astype(np.uint8))
        cv2.imwrite(save_name_label_image, im_label)
        cv2.imwrite(save_name_score, self.score.astype(np.uint8))
