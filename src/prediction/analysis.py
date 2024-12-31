"""Module containing functions to analyze the prediction results."""
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import cv2

from src.data.dataset_handler import resize_images
from src.data.analysis import get_highlighted_roi_by_mask

def plot_image_prediction(image: np.ndarray, mask: np.ndarray,
                          resize_shape: Optional[Tuple[int, int]] = None
                          ) -> None:
    """
    Plot an image, the predicted segmentation mask and the highlighted
    ROI of the segmentation mask.

    Parameters
    ----------
    image : ndarray
        Numpy array representing the input image.
    masks : np.ndarray
        Numpy array representing the segmentation mask of the image.
    resize_shape : (int, int), optional
        The size used to reshape images before plotting.
        By default None.
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 8))

    if resize_shape is not None:
        [image] = resize_images([image], resize_shape)
        [mask] = resize_images([mask], resize_shape)

    # Plot color image.
    ax = axes[0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Plot mask.
    ax = axes[1]
    ax.imshow(mask, cmap='gray', vmin=0., vmax=1.)
    ax.axis('off')
    legend_elements = [
        Patch(facecolor='w', edgecolor='black',label='Fire mask')]
    ax.legend(handles=legend_elements, loc='upper left')

    # Plot highlighted mask over the color image.
    ax = axes[2]
    highlighted_roi = get_highlighted_roi_by_mask(
        image, mask, highlight_channel='red')
    ax.imshow(cv2.cvtColor(highlighted_roi, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    legend_elements = [
        Patch(facecolor='r', edgecolor='black', label='Fire ROI')]
    ax.legend(handles=legend_elements, loc='upper left')
    title = 'An image along with the predicted fire mask and the highlighted '
    'segmentation'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_predicted_mask(mask: np.ndarray, save_path: str, resize_shape: Optional[Tuple[int, int]] = None) -> None:
    """
    Save the predicted mask to the specified path as a PNG image.
    
    Parameters
    ----------
    mask : np.ndarray
        The predicted mask to be saved.
    save_path : str
        The path where the mask will be saved.
    resize_shape : tuple, optional
        The size to which the mask will be resized before saving. If None, the mask is saved as is.
    """
    # Resize the mask if resize_shape is provided
    # Check if the mask is valid
    if mask is None:
        raise ValueError("The predicted mask is None, unable to save.")
    
    #if resize_shape:
    #    if len(mask.shape) < 2:
    #        raise ValueError(f"Mask has incorrect shape {mask.shape} for resizing.")
    #    mask = cv2.resize(mask, resize_shape)

    # Ensure the mask is in a suitable format for saving as an image (0-255)
    mask = (mask * 255).astype(np.uint8)

    # Save the mask using OpenCV (as PNG)
    cv2.imwrite(save_path, mask)
    print(f"Mask saved to {save_path}")