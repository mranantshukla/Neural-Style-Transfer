"""
VGG19 model for feature extraction in Neural Style Transfer.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


# Content layer for feature extraction
CONTENT_LAYER = 'block4_conv2'

# Style layers for feature extraction
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]


def get_vgg19_model():
    """
    Load VGG19 model pretrained on ImageNet.
    
    Returns:
        Model: VGG19 model with top layers removed
    """
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg


def get_feature_extractor():
    """
    Create a feature extractor model that outputs content and style features.
    
    Returns:
        Model: Keras model that outputs content and style layer activations
    """
    vgg = get_vgg19_model()
    
    # Get outputs from content and style layers
    content_output = vgg.get_layer(CONTENT_LAYER).output
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    
    # Combine all outputs
    model_outputs = [content_output] + style_outputs
    
    # Create model that outputs these features
    model = Model(vgg.input, model_outputs)
    
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False
    
    return model


def get_num_content_layers():
    """Return the number of content layers."""
    return 1


def get_num_style_layers():
    """Return the number of style layers."""
    return len(STYLE_LAYERS)

