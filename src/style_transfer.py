"""
Neural Style Transfer implementation using VGG19.
Main optimization loop for generating stylized images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.model import get_feature_extractor, get_num_content_layers, get_num_style_layers
from src.losses import gram_matrix, compute_total_loss
from src.utils import load_image, preprocess_image, deprocess_image


class StyleTransfer:
    """
    Neural Style Transfer class that applies artistic style to content images.
    """
    
    def __init__(self, content_image_path, style_image_path, max_size=400):
        """
        Initialize the StyleTransfer with content and style images.
        
        Args:
            content_image_path (str): Path to content image
            style_image_path (str): Path to style image
            max_size (int): Maximum dimension for image resizing
        """
        self.max_size = max_size
        
        # Load and preprocess images
        self.content_img = load_image(content_image_path, max_size)
        self.style_img = load_image(style_image_path, max_size)
        
        # Preprocess for VGG19
        self.content_array = preprocess_image(self.content_img)
        self.style_array = preprocess_image(self.style_img)
        
        # Convert to tensors
        self.content_tensor = tf.constant(self.content_array)
        self.style_tensor = tf.constant(self.style_array)
        
        # Load VGG19 feature extractor
        self.model = get_feature_extractor()
        
        # Extract features
        self.content_features = None
        self.gram_style_features = None
        
        self.num_content_layers = get_num_content_layers()
        self.num_style_layers = get_num_style_layers()
    
    def extract_features(self):
        """Extract content and style features from images."""
        # Extract content features
        content_outputs = self.model(self.content_tensor)
        content_features = [content_layer[0] for content_layer in 
                           content_outputs[:self.num_content_layers]]
        
        # Extract style features
        style_outputs = self.model(self.style_tensor)
        style_features = [style_layer[0] for style_layer in 
                         style_outputs[self.num_content_layers:]]
        
        # Compute Gram matrices for style features
        gram_style_features = [gram_matrix(style_feature) 
                              for style_feature in style_features]
        
        self.content_features = content_features
        self.gram_style_features = gram_style_features
    
    def generate(self, epochs=10, steps_per_epoch=5, learning_rate=5.0,
                 content_weight=1e4, style_weight=1e-2, tv_weight=0.0):
        """
        Generate stylized image using gradient descent optimization.
        
        Args:
            epochs (int): Number of training epochs
            steps_per_epoch (int): Number of optimization steps per epoch
            learning_rate (float): Learning rate for Adam optimizer
            content_weight (float): Weight for content loss
            style_weight (float): Weight for style loss
            tv_weight (float): Weight for total variation loss
            
        Returns:
            numpy.ndarray: Generated stylized image (0-255, RGB, uint8)
        """
        # Extract features if not already done
        if self.content_features is None:
            self.extract_features()
        
        # Initialize generated image from content image
        generated_image = tf.Variable(self.content_array, dtype=tf.float32)
        
        # Setup optimizer
        optimizer = Adam(learning_rate=learning_rate)
        
        # Track best result and loss history
        best_loss = float('inf')
        best_img = None
        loss_history = {
            'total_loss': [],
            'content_loss': [],
            'style_loss': [],
            'epochs': []
        }
        
        # Optimization loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                with tf.GradientTape() as tape:
                    # Get model outputs for generated image
                    model_outputs = self.model(generated_image)
                    
                    # Compute loss
                    total_loss, content_score, style_score, tv_score = compute_total_loss(
                        model_outputs,
                        self.content_features,
                        self.gram_style_features,
                        content_weight,
                        style_weight,
                        tv_weight,
                        self.num_content_layers,
                        self.num_style_layers
                    )
                
                # Compute gradients
                gradients = tape.gradient(total_loss, generated_image)
                
                # Apply gradients
                optimizer.apply_gradients([(gradients, generated_image)])
                
                # Clip values to valid range for VGG19 preprocessing
                # VGG19 preprocessing expects values in range [-103.939, 255 - 103.939]
                clipped = tf.clip_by_value(
                    generated_image,
                    -103.939,
                    255.0 - 103.939
                )
                generated_image.assign(clipped)
            
            # Track losses
            current_loss = total_loss.numpy()
            current_content_loss = content_score.numpy()
            current_style_loss = style_score.numpy()
            
            loss_history['total_loss'].append(current_loss)
            loss_history['content_loss'].append(current_content_loss)
            loss_history['style_loss'].append(current_style_loss)
            loss_history['epochs'].append(epoch + 1)
            
            # Track best result
            if current_loss < best_loss:
                best_loss = current_loss
                best_img = deprocess_image(generated_image.numpy())
            
            # Print progress (optional, can be removed for web app)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.2f}, "
                  f"Content: {current_content_loss:.2f}, Style: {current_style_loss:.2f}")
        
        # Return best image and loss history
        if best_img is not None:
            return best_img, loss_history
        else:
            return deprocess_image(generated_image.numpy()), loss_history

