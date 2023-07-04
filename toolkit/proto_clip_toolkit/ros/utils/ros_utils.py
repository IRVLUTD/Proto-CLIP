import time
import rospy
import numpy as np
import tf.transformations as tra
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from transforms3d.quaternions import mat2quat, quat2mat


def convert_rosqt_to_standard(pose_ros):
    """Converts (posn, x,y,z,w) quat to (posn, w,x,y,z) quat"""
    posn = pose_ros[:3]
    ros_qt = pose_ros[3:]
    quat = [ros_qt[-1], ros_qt[0], ros_qt[1], ros_qt[2]]
    return [*posn, *quat]


def convert_standard_to_rosqt(pose_s):
    """Converts (posn, w,x,y,z) quat to ROS format (posn, x,y,z,w) quat"""
    posn = pose_s[:3]
    q_s = pose_s[3:]
    quat = [q_s[1], q_s[2], q_s[3], q_s[0]]
    return [*posn, *quat]


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def ros_pose_to_rt(pose):
    qarray = [0, 0, 0, 0]
    qarray[0] = pose.orientation.x
    qarray[1] = pose.orientation.y
    qarray[2] = pose.orientation.z
    qarray[3] = pose.orientation.w

    t = [0, 0, 0]
    t[0] = pose.position.x
    t[1] = pose.position.y
    t[2] = pose.position.z

    return ros_qt_to_rt(qarray, t)


def rt_to_ros_pose(pose, rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]

    return pose


def rt_to_ros_qt(rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    return quat, trans


def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True):
    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    return X


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    time_start = time.time()
    while time.time() - time_start < 3:
        try:
            init_trans, init_rot = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            break
        except Exception as e:
            if first_time:
                print(str(e))
            init_trans = np.array([0, 0, 0])
            init_rot = np.array([0, 0, 0, 1])
            continue

    # print('got relative pose between {} and {}'.format(source_frame, target_frame))
    return ros_qt_to_rt(init_rot, init_trans)


pallete = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0.5, 0.5, 0],
    [1, 1, 1],
    [1, 1, 1],
    [0, 1, 1],
]


def map_seg_image(image):
    image = np.squeeze(image)
    output_image = [
        np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for _ in range(3)
    ]
    for i, color in enumerate(pallete):
        mask = image == (i + 1)
        for j in range(3):
            output_image[j][mask] = color[2 - j] * 255
    for i in range(3):
        output_image[i] = np.expand_dims(output_image[i], -1)

    return np.concatenate(output_image, -1)


def create_gripper_marker_message(
    frame_id,
    namespace,
    mesh_resource,
    color,
    lifetime=True,
    mesh_use_embedded_materials=True,
    marker_id=0,
    frame_locked=False,
):
    marker = Marker()
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.ns = namespace
    if lifetime:
        marker.lifetime = rospy.Duration(0.2)
    marker.frame_locked = frame_locked
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = mesh_resource
    marker.mesh_use_embedded_materials = mesh_use_embedded_materials

    return marker


def publish_grasps(publisher, frame_id, grasps, color_alpha, scores=None):
    markers = MarkerArray()
    for i, g in enumerate(grasps):
        if scores is None:
            x = float(i) / len(grasps)
        else:
            x = scores

        color = [1 - x, x, 0, color_alpha]
        marker = create_gripper_marker_message(
            # marker = create_axis_marker_message (
            frame_id=frame_id,
            namespace="hand",
            mesh_resource="package://grasping_vae/panda_gripper.obj",
            color=color,
            marker_id=i,
        )
        pos = tra.translation_from_matrix(g)
        quat = tra.quaternion_from_matrix(g)
        marker.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
        markers.markers.append(marker)

    # rospy.loginfo('markers length {}'.format(len(markers.markers)))
    publisher.publish(markers)
