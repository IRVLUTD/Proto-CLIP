import rospy
import sys
import argparse
import numpy as np
import ros_numpy
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from proto_clip_toolkit.ros.utils import SegImageListener, ProtoClipClassifier, crop_object_images


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--adapter', dest='adapter', help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--memory_bank_v_path', dest='memory_bank_v_path', help='path to the visual embeddings memory bank', required=True)
    parser.add_argument('--memory_bank_t_path', dest='memory_bank_t_path', help='path to the textual embeddings memory bank', required=True)
    parser.add_argument('--adapter_weights_path', dest='adapter_weights_path', help='path to the weights of the pretrained-query-adapter', required=True)
    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = get_arguments()

    #Start the Node
    rospy.init_node("proto_clip_result_pub")

    bridge = CvBridge()
    exp_dir = "testing_grap"
    listener = SegImageListener(data_dir=exp_dir)
    predictions_pub = rospy.Publisher("/proto_clip_pred", Image, queue_size=10)
    proto_clip = ProtoClipClassifier(args)
    
    step = 0

    #We run the preictions node every 5 seconds.
    while True:
        # Listen for the segmentation label until we receive a good segmentation.
        while True:
            label = listener.label
            xyz_image = listener.xyz_image
            camera_pose = listener.camera_pose
            rgb_image = listener.im
            bbox = listener.bbox
            score = listener.score

            depth_stamp = listener.depth_frame_stamp

            if bbox is None:
                rospy.loginfo("No object segmented")
                continue
            # filter objects
            index = bbox[:, 0] < 1.5
            bbox = bbox[index, :]

            print(f"{bbox.shape[0]} objects segmented")
            listener.save_data(step)
            break
        
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        cropped_images, mask_ids = crop_object_images(label, rgb_image)

        top_k_classes, top_k_probs = proto_clip.classify_objects(cropped_images, rgb_image=rgb_image, log=False)
        img, top_k_text = proto_clip.draw_image_with_top_k_images(cropped_images, top_k_classes, top_k_probs)

        label_msg = ros_numpy.msgify(Image, np.array(img), encoding="rgb8")
        predictions_pub.publish(label_msg)

        rospy.sleep(5)