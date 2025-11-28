import os
import sys
import scipy.io
import numpy as np
import tensorflow as tf
from PIL import Image 

# Use Keras VGG19 instead of manual .mat file loading for TF2 compatibility
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    # Mean pixel values for VGG (BGR order)
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    # Note: We use the Keras internal weights now, so the .mat file path is optional/unused
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' 
    STYLE_IMAGE = 'images/monet_800600.jpg' 
    CONTENT_IMAGE = 'images/louvre.jpg' 
    OUTPUT_DIR = 'output/'

def load_vgg_model(path=None):
    """
    Returns a Keras model for the purpose of 'painting' the picture.
    Replaced manual .mat loading with standard Keras VGG19 for TF2 compatibility.
    """
    # Load VGG19 with ImageNet weights, exclude top (classification) layers
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # Freeze weights so they don't change during training
    
    # We need to access internal layers by name to get content/style features
    # These names match the Keras VGG19 layer naming convention
    content_layers = ['block4_conv2'] 
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1', 
        'block4_conv1', 
        'block5_conv1'
    ]
    
    # Get the output tensors for the specific layers we need
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    
    # Create a new model that takes an image and outputs the feature maps
    model = Model(inputs=vgg.input, outputs=outputs)
    
    return model

def generate_noise_image(content_image, noise_ratio=CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    # Generate a random noise_image
    noise_image = np.random.uniform(
        -20, 20, 
        (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)
    ).astype('float32')
    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    # Reshape image to match expected input of VGG19
    image = np.reshape(image, ((1,) + image.shape))
    
    # Subtract the mean to match the expected input of VGG19
    image = image - CONFIG.MEANS
    
    return image

def save_image(path, image):
    """
    Save the image using PIL (replaces deprecated scipy.misc.imsave)
    """
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS
    
    # Clip to valid pixel range 0-255
    if image.ndim == 4:
        image = image[0]
        
    image = np.clip(image, 0, 255).astype('uint8')
    
    # Save using PIL
    img = Image.fromarray(image)
    img.save(path)