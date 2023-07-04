from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
from proto_clip_toolkit.ros.utils import ProtoClipClassifier
import argparse
import matplotlib.pyplot as plt

#The mapping of each class idx to their set
test_data_set_idx_mapping = {
    1: [2, 6, 15, 26],
    2: [0, 13, 16, 18],
    3: [3, 14, 17, 24],
    4: [7, 10, 25, 31],
    5: [4, 5, 11, 29],
    6: [8, 19, 20, 23],
    7: [1, 12, 22, 27],
    8: [9, 21, 28, 30]
}

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--data_dir', dest='data_dir', help="directory containing the fewsol images", required=True)
    parser.add_argument('--adapter', dest='adapter', help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--adapter', dest='adapter', help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--memory_bank_v_path', dest='memory_bank_v_path', help='path to the visual embeddings memory bank', required=True)
    parser.add_argument('--memory_bank_t_path', dest='memory_bank_t_path', help='path to the textual embeddings memory bank', required=True)
    parser.add_argument('--adapter_weights_path', dest='adapter_weights_path', help='path to the weights of the pretrained-query-adapter', required=True)
    args = parser.parse_args()
    return args

test_data_set_idx_mapping = {
    1: [2, 6, 15, 26],
    2: [0, 13, 16, 18],
    3: [3, 14, 17, 24],
    4: [7, 10, 25, 31],
    5: [4, 5, 11, 29],
    6: [8, 19, 20, 23],
    7: [1, 12, 22, 27],
    8: [9, 21, 28, 30]
}

if __name__=="__main__":

    args = get_arguments()
    classifier = ProtoClipClassifier(args)

    f = open(args.splits_path)
    test_split_json = json.load(f)
    test_data = np.array(test_split_json["test"])
    output = []

    figure, axis = plt.subplots(8, 4)
    os.makedirs("./set-images", exist_ok=True)

    for (set_idx, set_elements) in test_data_set_idx_mapping.items():
        set_image_data = test_data[set_elements]

        ground_truth_names = set_image_data[:, 2]
        ground_truth_names = [name.replace("_", " ") for name in ground_truth_names]
        
        images = [f"{args.data_dir}/{location}" for location in set_image_data[:, 0]]
        pil_images = [Image.open(value) for value in images]
        for col_idx in range(len(images)):
            axis[set_idx-1, col_idx].imshow(pil_images[col_idx])

        top_k_classes, top_k_probs = classifier.classify_objects(pil_images, log=False)
        
        images = [cv2.imread(name) for name in images]
        draw_image, text_data = classifier.draw_image_with_top_k_images(images, top_k_classes, top_k_probs, ground_truth_names)
        draw_image.save(f"./set-images/set_{set_idx}.png", quality=200)
        
        text_data = [f"Set {set_idx}"] + text_data
        output.extend(text_data)
        # plt.imshow(draw_image)
        # plt.savefig(f"./images/set_{set_idx}.pdf", dpi=1000)
    plt.show()
    print("\n".join(output))

