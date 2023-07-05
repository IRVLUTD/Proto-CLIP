import rospy
import sys
import argparse
import numpy as np
import ros_numpy
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from proto_clip_toolkit.asr import transcribe_with_verb_and_noun_matching
from proto_clip_toolkit.pos import VerbAndNounTagger
from proto_clip_toolkit.ros.utils import crop_object_images, ProtoClipClassifier, SegImageListener


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--adapter', dest='adapter', help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--memory_bank_v_path', dest='memory_bank_v_path', help='path to the visual embeddings memory bank', required=True)
    parser.add_argument('--memory_bank_t_path', dest='memory_bank_t_path', help='path to the textual embeddings memory bank', required=True)
    parser.add_argument('--adapter_weights_path', dest='adapter_weights_path', help='path to the weights of the pretrained-query-adapter', required=True)
    parser.add_argument('--asr_verbs_path', dest='asr_verbs_path', help='path to dictionary of accepted verbs(actions)', required=True)
    parser.add_argument('--asr_nouns_path', dest='asr_nouns_path', help='path to dictionary of accepted nouns/objects', required=True)
    parser.add_argument('--asr_config_path', dest='asr_config_path', help='path to json containing configuration for asr', required=True)
    parser.add_argument('--splits_path', dest='splits_path', help='path to json containing split configuration for asr', required=True)
    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = get_arguments()

    #Start the Node
    rospy.init_node("proto_clip_with_asr")

    bridge = CvBridge()
    exp_dir = "testing_grap"
    listener = SegImageListener(data_dir=exp_dir)
    label_pub = rospy.Publisher("/selected_seg_label", Image, queue_size=10)
    score_pub = rospy.Publisher("/selected_seg_score", Image, queue_size=10)

    proto_clip = ProtoClipClassifier(args)
    pos_tagger = VerbAndNounTagger(args.asr_verbs_path, args.asr_nouns_path)

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
            break
        
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        cropped_images, mask_ids = crop_object_images(label, rgb_image)

        top_k_classes, top_k_probs = proto_clip.classify_objects(cropped_images, rgb_image=rgb_image)
        detected_classes = [x[0] for x in top_k_classes]
        print("The top-1 detected classes are", detected_classes)
        
        print("Starting transcription with whisper")
        spoken_action, spoken_noun = transcribe_with_verb_and_noun_matching(args.asr_config_path, pos_tagger)


        matching_k_idxes = [row.index(spoken_noun) if spoken_noun in row else -1 for row in top_k_classes]
        matching_idxes_probs = []
        chosen_img_idx, chosen_img_prob = None, float('-inf')

        for img_idx in range(len(matching_k_idxes)):
            if matching_k_idxes[img_idx]!=-1 and top_k_probs[img_idx][matching_k_idxes[img_idx]] > chosen_img_prob:
                chosen_img_idx = img_idx
                chosen_img_prob = top_k_probs[img_idx][matching_k_idxes[img_idx]]

        if chosen_img_idx==None:
            print("The spoken word is not present in any prediction")
            rospy.spin()

        selected_mask_id = mask_ids[chosen_img_idx]

        selected_obj_label = (label == selected_mask_id).astype(np.uint8)

        label_msg = ros_numpy.msgify(Image, selected_obj_label, encoding="mono8")
        score_msg = ros_numpy.msgify(Image, score, encoding="mono8")

        label_msg.header.stamp = depth_stamp
        score_msg.header.stamp = depth_stamp
        
        count = 0

        #Publish the message 10 times since at times the graping node fails because it cannot detect the 
        while count!=10:
            label_pub.publish(label_msg)
            score_pub.publish(score_msg)
            
            count += 1
        
        _next_step = input("Proceed to recognize next object y or n:")
        print(_next_step)
        while _next_step!="y" and _next_step!="n":
            _next_step = input("Proceed to recognize next object y or n:")

        if _next_step=="y":
            continue
        else:
            print("Program is done")
            break