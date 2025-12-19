"""
Loss functions for Neural Style Transfer.
Includes content loss, style loss (Gram matrix), and total variation loss.
"""

import tensorflow as tf


def gram_matrix(input_tensor):
    """
    Compute the Gram matrix of an input tensor.
    The Gram matrix captures style information by computing correlations
    between feature maps.
    
    Args:
        input_tensor: Tensor of shape (height, width, channels)
        
    Returns:
        Tensor: Gram matrix of shape (channels, channels)
    """
    # Get the number of channels
    channels = int(input_tensor.shape[-1])
    
    # Reshape to (height * width, channels)
    a = tf.reshape(input_tensor, [-1, channels])
    
    # Get the number of spatial locations
    n = tf.shape(a)[0]
    
    # Compute Gram matrix: a^T * a
    gram = tf.matmul(a, a, transpose_a=True)
    
    # Normalize by number of spatial locations
    return gram / tf.cast(n, tf.float32)


def content_loss(base_content, target):
    """
    Compute content loss as mean squared error between feature representations.
    
    Args:
        base_content: Feature representation of generated image
        target: Feature representation of content image
        
    Returns:
        Tensor: Scalar content loss value
    """
    return tf.reduce_mean(tf.square(base_content - target))


def style_loss(base_style, gram_target):
    """
    Compute style loss as mean squared error between Gram matrices.
    
    Args:
        base_style: Feature representation of generated image
        gram_target: Gram matrix of style image features
        
    Returns:
        Tensor: Scalar style loss value
    """
    # Compute Gram matrix for generated image
    gram_style = gram_matrix(base_style)
    
    # Compute MSE between Gram matrices
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def total_variation_loss(image):
    """
    Compute total variation loss to encourage spatial smoothness.
    This helps reduce noise and artifacts in the generated image.
    
    Args:
        image: Image tensor of shape (batch, height, width, channels)
        
    Returns:
        Tensor: Scalar total variation loss value
    """
    # Compute differences between adjacent pixels
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    # Sum of squared differences
    return tf.reduce_mean(tf.square(x_deltas)) + tf.reduce_mean(tf.square(y_deltas))


def compute_total_loss(model_outputs, content_features, gram_style_features,
                      content_weight, style_weight, tv_weight=0.0,
                      num_content_layers=1, num_style_layers=5):
    """
    Compute the total loss for neural style transfer.
    
    Args:
        model_outputs: List of model outputs (content + style features)
        content_features: Target content features
        gram_style_features: Target style Gram matrices
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        tv_weight: Weight for total variation loss (optional)
        num_content_layers: Number of content layers
        num_style_layers: Number of style layers
        
    Returns:
        tuple: (total_loss, content_score, style_score, tv_score)
    """
    # Split model outputs into content and style features
    content_output_features = model_outputs[:num_content_layers]
    style_output_features = model_outputs[num_content_layers:]
    
    # Compute content loss
    content_score = 0
    for target_content, comb_content in zip(content_features, content_output_features):
        # Remove batch dimension if present
        if len(comb_content.shape) == 4:
            comb_content = comb_content[0]
        content_score += content_loss(comb_content, target_content)
    
    # Normalize by number of content layers
    content_score *= content_weight / num_content_layers
    
    # Compute style loss
    style_score = 0
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        # Remove batch dimension if present
        if len(comb_style.shape) == 4:
            comb_style = comb_style[0]
        style_score += style_loss(comb_style, target_style)
    
    # Normalize by number of style layers
    style_score *= style_weight / num_style_layers
    
    # Compute total variation loss (if weight > 0)
    tv_score = 0
    if tv_weight > 0:
        # Get the generated image from model inputs (approximate)
        # Note: In practice, we'll pass the image separately for TV loss
        pass
    
    # Total loss
    total_loss = content_score + style_score + tv_score
    
    return total_loss, content_score, style_score, tv_score

