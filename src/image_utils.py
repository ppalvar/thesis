import os
import random
import hashlib
from typing import List, Tuple

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

IMG_DTYPE = np.float32


class ImageManager:
    """
    A class to manage and retrieve grayscale images from a directory and its subdirectories.

    Args:
        directory (str): Path to the root directory containing images.

    Attributes:
        image_paths (list): Valid paths to readable grayscale images (JPG/PNG) found in the directory tree.
    """

    def __init__(self, directory):
        self.directory = directory
        self.image_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, file)
                    if self._is_readable(full_path):
                        self.image_paths.append(full_path)

    def _is_readable(self, path):
        """Check if an image can be read as grayscale."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img is not None

    def get_all_images(self):
        """
        Retrieve all valid images as OpenCV grayscale matrices.

        Returns:
            list: List of numpy arrays representing grayscale images
        """
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in self.image_paths]

    def get_random_sample(self, k):
        """
        Get random sample of images.

        Args:
            k (int): Number of images to sample

        Returns:
            list: List of numpy arrays. Returns empty list if no images available
        """
        if not self.image_paths:
            return []

        k = min(k, len(self.image_paths))
        sampled_paths = random.sample(self.image_paths, k)
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in sampled_paths]

    def get_random_sample_lambdas(self, k):
        """
        Get lambdas that load images on demand.

        Args:
            k (int): Number of lambdas to sample

        Returns:
            list: List of callables that return numpy arrays when invoked.
                  Returns empty list if no images available
        """
        if not self.image_paths:
            return []

        k = min(k, len(self.image_paths))
        sampled_paths = random.sample(self.image_paths, k)

        # Use closure to capture current path value
        return [
            (lambda p: lambda: cv2.imread(p, cv2.IMREAD_GRAYSCALE))(path)
            for path in sampled_paths
        ]


def array_hash(arr):
    """Generate a SHA256 hash for a NumPy array."""
    hash_obj = hashlib.sha256()

    shape = np.array(arr.shape, dtype=np.int64)
    hash_obj.update(shape.tobytes())

    hash_obj.update(arr.dtype.name.encode("utf-8"))

    hash_obj.update(arr.tobytes())

    return hash_obj.hexdigest()


def normalize_img(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Normalizes the input image array to the range [0, 1].
    Parameters:
        img (np.ndarray): Input image as a NumPy array.
    Returns:
        Tuple[np.ndarray, int, int]: A tuple containing:
            - The normalized image as a NumPy array.
            - The minimum value found in the original image.
            - The maximum value found in the original image after subtracting the minimum.
    """
    min_val = np.min(img)
    img = img - min_val

    max_val = np.max(img)
    img = img / max_val

    return img, min_val, max_val


def denormalize_img(img: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """
    Denormalizes an image array from a normalized range [0, 1] back to the original range [min_val, max_val].
    Parameters:
        img (np.ndarray): The normalized image array with values in the range [0, 1].
        min_val (int): The minimum value of the original image range.
        max_val (int): The maximum value of the original image range.
    Returns:
        np.ndarray: The denormalized image array with values in the range [min_val, max_val].
    """
    img = img * max_val
    img = img + min_val

    return img


def load_nii(nii_file: str) -> np.ndarray:
    """
    Loads a NIfTI (.nii) file and returns its image data as a NumPy array.
    Args:
        nii_file (str): Path to the NIfTI file to be loaded.
    Returns:
        np.ndarray: Image data extracted from the NIfTI file, cast to IMG_DTYPE.
    Raises:
        AssertionError: If the provided file path does not exist.
    """
    assert os.path.isfile(nii_file), f"The provided file {nii_file} was not found."

    n = nib.load(nii_file)  # type: ignore

    data = n.get_fdata()  # type: ignore

    return data.astype(IMG_DTYPE)  # type: ignore


def get_nii_as_list(nii: np.ndarray) -> List[np.ndarray]:
    """
    Splits a 3D NumPy array (typically loaded from a .nii file) into a list of 2D slices along the last axis.
    Args:
        nii (np.ndarray): A 3D NumPy array of shape (H, W, N), where N is the number of slices.
    Returns:
        List[np.ndarray]: A list containing N 2D NumPy arrays, each of shape (H, W), corresponding to the slices along the last axis.
    Raises:
        AssertionError: If the input array does not have exactly 3 dimensions.
    """
    assert nii.ndim == 3, "The provided .nii content must be of shape (H, W, N)."

    result = [nii[..., i] for i in range(nii.shape[-1])]

    return result


def animate_nii(data: np.ndarray):
    """
    Animates a 3D NumPy array (Height x Width x Frames) as a grayscale image sequence.
    Parameters
    ----------
    data : np.ndarray
        A 3D NumPy array with shape (Height, Width, Frames), where each frame is displayed sequentially.
    Raises
    ------
    AssertionError
        If the input array is not 3-dimensional.
    Notes
    -----
    This function temporarily switches the matplotlib backend to 'TkAgg' for animation,
    and restores the original backend after completion. Each frame is displayed for 0.15 seconds.
    The animation window is closed automatically after all frames are shown.
    """
    assert data.ndim == 3, (
        "The input for animation must be a (Height x Width x Frames) array"
    )

    _, _, frames = data.shape

    backend_bak = plt.get_backend()
    plt.switch_backend("TkAgg")

    fig, ax = plt.subplots(tight_layout=True)
    ax.axis("off")

    img = ax.imshow(data[..., 0], cmap="gray")
    plt.show(block=False)
    fig.canvas.draw()

    for i in range(frames):
        img.set_data(data[..., i])

        fig.canvas.draw()

        fig.canvas.flush_events()

        plt.pause(0.15)

    plt.close()
    plt.switch_backend(backend_bak)


def apply_window(data, level, width):
    """
    Applies a windowing operation to image data, commonly used in medical imaging to enhance contrast.

    Parameters:
        data (np.ndarray): Input image data as a NumPy array.
        level (float): The center of the window (window level).
        width (float): The width of the window (window width).

    Returns:
        np.ndarray: The windowed and normalized image data as an 8-bit unsigned integer array (values scaled to 0-255).
    """
    window_min = level - width / 2
    window_max = level + width / 2
    clipped = np.clip(data, window_min, window_max)
    normalized = (clipped - window_min) / width * 255  # Scale to 0-255
    return normalized.astype(np.uint8)


def get_windowed_image(ct_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies brain and bone windowing presets to a CT image.
    Parameters:
        ct_data (np.ndarray): The input CT image data as a NumPy array.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The CT image windowed with the brain preset.
            - The CT image windowed with the bone preset.
    Notes:
        This function uses predefined window level and width settings for brain and bone tissue visualization.
        The `apply_window` function is expected to handle the actual windowing operation.
    """
    presets = {
        "brain": {"level": 40, "width": 120},
        "bone": {"level": 700, "width": 3200},
    }

    brain_windowed = apply_window(ct_data, **presets["brain"])
    bone_windowed = apply_window(ct_data, **presets["bone"])

    return brain_windowed, bone_windowed
