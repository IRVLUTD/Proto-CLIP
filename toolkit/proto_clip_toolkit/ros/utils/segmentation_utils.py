import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):
    """Visualize segmentations nicely. Based on code from:
    https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

    @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
    @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
    @param nc: total number of colors. If None, this will be inferred by masks
    """
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)

    # Draw color masks
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = 0.4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = masks == i

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)

    # Draw mask contours
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = 0.4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = masks == i

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=False,
                    facecolor=color_mask,
                    edgecolor="w",
                    linewidth=1.2,
                    alpha=0.5,
                )
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255, 255, 255), 2)

    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image
