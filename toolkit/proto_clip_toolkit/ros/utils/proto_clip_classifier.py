import os
import yaml
import sys
import json

import time

import numpy as np
import random
import torch
 

sys.path.append("../../")
from utils import *

import clip
from PIL import Image, ImageDraw, ImageFont
from .image_utils import RealWorldDataset
from proto_clip_toolkit.utils import load_pretrained_mb_and_adapters, pre_load_features_without_cache

class ProtoClipClassifier:
    def __init__(self, args):
        assert (os.path.exists(args.config))

        self.cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        print("\nRunning configs.")
        print(self.cfg, "\n")

        self.clip_model, self.preprocess = clip.load(self.cfg['backbone'])
        self.clip_model.eval()

        seed = get_seed()
        random.seed(seed)
        np.random.seed(seed)
        g = torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.test_bs = 1
        self.n_workers = 1
        self.class_id_mapping = {}

        self.parse_splits_file(args.splits_path)
        self._load_trained_models_and_embeddings(args)
    
    def _load_trained_models_and_embeddings(self, args):
        """Loads the pretrained embeddings and adapter model using the location provided by the user."""

        with torch.no_grad():
            # CLIP

            embeddings_v, embeddings_t, self.adapter = load_pretrained_mb_and_adapters(
                adapter_type=args.adapter if args.adapter else self.cfg["adapter"], 
                memory_bank_v_path=args.memory_bank_v_path,
                memory_bank_t_path=args.memory_bank_t_path,
                adapter_weights_path=args.adapter_weights_path
            )

            NxK, ndim = embeddings_v.shape
            K = self.cfg['shots']
            N = NxK//K

            zs_imgs = embeddings_v.view(-1, K, ndim)
            zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
            self.z_img_proto = zs_imgs.mean(dim=1)
            self.z_img_proto = self.z_img_proto / self.z_img_proto.norm(dim=-1, keepdim=True)

            zs_text = embeddings_t
            self.z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)
        
    
    def parse_splits_file(self, config_path):
        """Returns the class id to class name mapping based on splits file"""
        f = open(config_path)
        data = json.load(f)

        for config_data in data["train"]:
            self.class_id_mapping[config_data[1]] = config_data[2]
    
    def draw_image_with_top_k_images(self, image_list, top_k_classes, top_k_probs, ground_truth_classes=None):

        """Returns the image containing the top-k predictions for each of the images in the image_list"""
        img = Image.new("RGB", (650, 325), (255, 255, 255))

        # Draw each of the images on the canvas first
        for img_idx in range(len(image_list)):
            x_coord = 40 + (img_idx % 2) * 300
            y_coord = 40 + (img_idx // 2) * 160 # Divide by 2 because we want only two images in one row. 

            imageRGB = image_list[img_idx]
            cropped_image = Image.fromarray(imageRGB)
            cropped_image = cropped_image.resize((100, 100))

            img.paste(cropped_image, box=(x_coord, y_coord))

        top_k_probs *= 100

        # Check if information about ground truth class is present to display it in bold.
        if ground_truth_classes!=None:
            ground_truth_idxes = [top_k_classes[idx].index(ground_truth_classes[idx]) if ground_truth_classes[idx] in top_k_classes[idx] else -1 for idx in range(len(ground_truth_classes))]
        else:
            ground_truth_idxes = [-1 for idx in range(len(top_k_classes))]


        # top_k_classes contains the entire text about top-k predictions for a single image.
        top_k_classes = [[f"{idx}. {top_k_classes[row_idx][idx-1]} ({round(top_k_probs[row_idx][idx-1].item(), 2)}%)" for idx in range(1, len(top_k_classes[row_idx]) + 1)] for row_idx in range(len(top_k_classes))]
        top_k_classes_text = ["\n".join(row) for row in top_k_classes]

        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("./fonts/Roboto-Regular.ttf", 12)
        bold_font = ImageFont.truetype("./fonts/Roboto-Bold.ttf", 12)

        # Draw the text on the image next to each of the cropped images. If ground truth class is not present for the 
        for txt_idx in range(len(top_k_classes)):
            x_coord = 150 + (txt_idx % 2) * 300
            start_y_coord = 40 + (txt_idx // 2) * 160 # Divide by 2 because we want only two images in one row. 
            next_row_pad = 20

            if ground_truth_classes!=None and ground_truth_idxes[txt_idx]==-1:
                top_k_classes_text[txt_idx] = "True class: " + ground_truth_classes[txt_idx] + "\n" + top_k_classes_text[txt_idx] 
                d.multiline_text((x_coord, start_y_coord - next_row_pad), "True class: " + ground_truth_classes[txt_idx], font=bold_font, fill="green")
            
            gt_k_idx = ground_truth_idxes[txt_idx]
            for class_idx in range(len(top_k_classes[txt_idx])):
                d.multiline_text((x_coord, start_y_coord + class_idx * next_row_pad), top_k_classes[txt_idx][class_idx], font= bold_font if class_idx==gt_k_idx else font, fill="black")
        
        return img, top_k_classes_text


    def classify_objects(self, cropped_images, log=True, rgb_image=None):
        """Returns the Proto-CLIP predictions along with their class names for the images."""
        test_dataset = RealWorldDataset(cropped_images, self.preprocess)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_bs, num_workers=self.n_workers, shuffle=False)

        test_features = pre_load_features_without_cache(self.clip_model, test_loader)
        
        with torch.no_grad():
            #Load test_featurs using UIOS rather than the given dataset.
            test_features = self.adapter(test_features)
            test_features = test_features / test_features.norm(dim=-1, keepdim=True)
            
            p = P(test_features, self.z_img_proto, self.z_text_proto, self.cfg['alpha'], self.cfg['beta'])

            top_k_class_probs = p.topk(k=self.cfg["top_k"], dim=1)[0]
            top_k_class_idxs = p.topk(k=self.cfg["top_k"], dim=1)[1]

            top_k_class_names = [[self.class_id_mapping[x.item()].replace("_", " ") for x in row] for row in top_k_class_idxs]
            
            if log:
                os.makedirs("./ros-demo-logs", exist_ok=True)
                pred_timestamp =  int(time.time())

                mat_file_details = {"rgb_image": rgb_image, "cropped_images": cropped_images, "top_k_classes": top_k_class_names, "top_k_probs": top_k_class_probs.cpu().numpy()}
                np.save(f"./ros-demo-logs/experiment_pred_{pred_timestamp}.npy", mat_file_details)

            return top_k_class_names, top_k_class_probs
