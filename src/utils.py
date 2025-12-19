"""
Image utility functions for loading, preprocessing, and saving images.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def get_image_list(folder_path):
    """
    Dynamically scan a folder and return list of image files.
    
    Args:
        folder_path (str): Path to the folder to scan
        
    Returns:
        list: List of image filenames found in the folder
    """
    if not os.path.exists(folder_path):
        return []
    
    image_files = []
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in IMAGE_EXTENSIONS:
            image_files.append(filename)
    
    return sorted(image_files)


def validate_image(file_path):
    """
    Validate that a file is a valid image.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def load_image(path, max_size=400):
    """
    Load and preprocess an image for VGG19.
    
    Args:
        path (str): Path to the image file
        max_size (int): Maximum dimension for resizing (maintains aspect ratio)
        
    Returns:
        PIL.Image: Loaded image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return img


def preprocess_image(img):
    """
    Preprocess image for VGG19 input.
    
    Args:
        img: PIL Image or numpy array
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for VGG19
    """
    if isinstance(img, Image.Image):
        img = kp_image.img_to_array(img)
    
    # Expand dimensions to add batch dimension
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Apply VGG19 preprocessing (BGR, zero-centered)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    
    return img


def deprocess_image(processed_img):
    """
    Convert preprocessed image back to displayable format.
    
    Args:
        processed_img: Preprocessed image array (can be tensor or numpy array)
        
    Returns:
        numpy.ndarray: Deprocessed image array (0-255, RGB, uint8)
    """
    # Convert tensor to numpy if needed
    if tf.is_tensor(processed_img):
        x = processed_img.numpy().copy()
    else:
        x = processed_img.copy()
    
    # Remove batch dimension if present
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    
    assert len(x.shape) == 3, (
        "Input to deprocessing image must be an image of "
        "dimension [1, height, width, channel] or [height, width, channel]"
    )
    
    # Reverse VGG19 preprocessing
    # VGG19 preprocessing: RGB -> BGR, then subtract [103.939, 116.779, 123.68]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # BGR -> RGB
    x = x[:, :, ::-1]
    
    # Clip to valid range and convert to uint8
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x


def save_image(img_array, path):
    """
    Save an image array to file.
    
    Args:
        img_array: Image array (numpy array or tensor)
        path (str): Output file path
    """
    # Deprocess if needed
    if isinstance(img_array, tf.Tensor) or (isinstance(img_array, np.ndarray) and img_array.dtype != np.uint8):
        img_array = deprocess_image(img_array)
    
    # Convert to PIL Image and save
    img = Image.fromarray(img_array)
    img.save(path, 'PNG')

