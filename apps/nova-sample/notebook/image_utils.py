import io
import os
import av
import base64
import requests
import tempfile
from io import BytesIO
from PIL import Image
from typing import Tuple
from IPython.display import display, Video, Image as IPythonImage



# Function to encode image from bytes or PIL.Image
def encode_image_base64(image, format="JPEG", max_size=(2000, 2000)):
    # If the input is not an instance of PIL.Image, open it
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Resize the image
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Save the image to buffer and encode as base64
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format=format)
    encoded_image = base64.b64encode(buffer.getvalue())
    return encoded_image.decode('utf-8')


# Function to convert byte data to PIL.Image object
def bytes_to_image(image_bytes: bytes):
    """
    Converts image byte data to a PIL.Image object.
    """
    return Image.open(BytesIO(image_bytes))


# Function to save image byte data to a file
def save_image_bytes(image_bytes: bytes, path: str):
    """
    Saves image byte data to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(image_bytes)



def create_outpainting_mask(source_image: Image,
                            target_width: int,
                            target_height: int,
                            position: Tuple[float, float] = (0.5, 0.5)) -> Image:
    """
    Creates a mask image for outpainting.
    
    Args:
        source_image (PIL.Image): Original image
        width (int): Target final width for expansion
        height (int): Target final height for expansion
        position: Placement position ratio of original image (x, y)
                 (0.5, 0.5): Center
                 (0.0, 0.0): Top left
                 (1.0, 1.0): Bottom right
                 (0.0, 0.5): Left center
        
    Returns:
        tuple: Mask image (PIL.Image), base64 encoded mask image (str))
    """
    # Original image size
    src_width, src_height = source_image.size
    
    # Raise error if new size is smaller than original
    if target_width < src_width or target_height < src_height:
        raise ValueError(
            f"Target dimensions({target_width}, {target_height}) must be larger than "
            f"source image dimensions({src_width}, {src_height})"
        )
        
    # Clamp position values between 0 and 1
    pos_x, pos_y = position
    pos_x = max(0.0, min(1.0, pos_x))
    pos_y = max(0.0, min(1.0, pos_y))
    
    # Fill the area where the original image will be placed with white
    # Calculate actual pixel coordinates
    paste_x = int((target_width - src_width) * pos_x)
    paste_y = int((target_height - src_height) * pos_y)
    
    # Create extended image
    BLACK = (255, 255, 255)
    WHITE = (0, 0, 0)
    extended_image = Image.new("RGB", (target_width, target_height), BLACK)
    extended_image.paste(source_image, (paste_x, paste_y))
    
    # Create mask image
    mask_image = Image.new("RGB", (target_width, target_height), BLACK)
    original_image_shape = Image.new(
        "RGB", (src_width, src_height), WHITE
    )
    mask_image.paste(original_image_shape, (paste_x, paste_y))
     
    return extended_image, mask_image




# Function to get byte data from an image
def get_image_bytes(image, format="JPEG", max_size=(1000, 1000)):
    """
    Converts an image to byte data.
    Args:
        image: PIL.Image object or image file path.
        format: Desired image format (e.g., "JPEG", "PNG").
        max_size: Maximum size of the image (width, height).
    Returns:
        bytes: Byte data of the image.
    """
    # If input is not a PIL.Image object, open the image.
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Resize the image
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Save the image to buffer
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return image_bytes


# Function to get image byte data from a URL
def get_image_bytes_from_url(img_url, format="JPEG", max_size=(1000, 1000)):
    """
    Fetches an image from a URL and returns its byte data.
    """
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()  # Raise exception for error responses
        return get_image_bytes(BytesIO(response.content), format, max_size)
    except Exception as e:
        print(f"Error fetching image from URL: {e}")
        return None


# Function to get image byte data from a file path
def get_image_bytes_from_file(file_path, format="JPEG", max_size=(1000, 1000)):
    """
    Reads an image from a file and returns its byte data.
    """
    try:
        return get_image_bytes(file_path, format, max_size)
    except Exception as e:
        print(f"Error reading image from file: {e}")
        return None




# Function to display an image using byte data
def display_image_bytes(image_bytes, format='PNG', height=200):
    """
    Displays an image from byte data.
    """
    if isinstance(image_bytes, bytes):
        display(IPythonImage(data=image_bytes, format=format, height=height))
    elif isinstance(image_bytes, list):
        for img_bytes in image_bytes:
            display(IPythonImage(data=img_bytes, format=format, height=height))


def display_video_bytes(video_bytes: bytes, width=800):
    temp_path = os.path.join(tempfile.gettempdir(), 'temp_video.mp4')
    with open(temp_path, 'wb') as f:
        f.write(video_bytes)
    video = Video(temp_path, embed=True, width=width, html_attributes="controls")
    display(video)

    try:
        os.remove(temp_path)
    except:
        pass