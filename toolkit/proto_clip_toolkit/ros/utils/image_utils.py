import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

class RealWorldDataset(Dataset):
    """Create a simple dataset object that returns the images"""
    def __init__(self, images, transform=None) -> None:
        super().__init__()

        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])

        if self.transform:
            img = self.transform(img)
        
        return img

def crop_object_images(label, rgb_image):
    """Returns images of individual objects from the segmentation image."""
    mask_ids = np.unique(label)
    cropped_objects = []

    if mask_ids[0]==0:
        mask_ids = mask_ids[1:]

    updated_mask_ids = []
    
    for mask_id in mask_ids:
        mask = label==mask_id

        x_indices, y_indices = np.nonzero(mask)

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)


        """We reject the cropped object if its size is very small."""
        if (y_max - y_min <= 5) or (x_max - x_min <= 5):
            continue

        cropped_image = rgb_image[x_min:x_max, y_min:y_max, :]

        # plt.imshow(cropped_image)
        # plt.show()

        updated_mask_ids.append(mask_id)
        cropped_objects.append(cropped_image)

    #Code to debug cropped images    
    # cv2.imshow("rgb_imag", rgb_image)

    return cropped_objects, updated_mask_ids