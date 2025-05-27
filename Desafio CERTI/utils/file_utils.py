import os
import cv2

def ensure_dir(directory='image_result'):
    """Ensure output directory exists."""
    if not os.path.isdir(directory):
        print(f'Creating directory {directory}')
        os.makedirs(directory)
    else:
        print(f'Directory {directory} already exists')

def load_image(filepath):
    """Load image from filepath."""
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {filepath}')
    return img

def save_image(filepath, image):
    """Save image to filepath."""
    cv2.imwrite(filepath, image)
    print(f'Image saved at {filepath}')
