"""
Flask web application for Neural Style Transfer.
Provides web interface for uploading/selecting images and generating stylized outputs.

Author: Anant Shukla (23113024)
Department: Civil Engineering, IIT Roorkee
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from src.style_transfer import StyleTransfer
from src.utils import get_image_list, validate_image, save_image

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_default_content_images():
    """Get list of default content images."""
    content_folder = 'contentimage'
    return get_image_list(content_folder)


def get_default_style_images():
    """Get list of default style images."""
    style_folder = 'style_images'
    return get_image_list(style_folder)


@app.route('/')
def index():
    """Render the main web interface."""
    return render_template('index.html')


@app.route('/api/default-images', methods=['GET'])
def get_default_images():
    """API endpoint to get list of available default images."""
    content_images = get_default_content_images()
    style_images = get_default_style_images()
    
    return jsonify({
        'content_images': content_images,
        'style_images': style_images
    })


@app.route('/api/generate', methods=['POST'])
def generate_stylized_image():
    """
    Generate stylized image from content and style images.
    Accepts either uploaded files or default image names.
    """
    try:
        # Get parameters
        epochs = int(request.form.get('epochs', 10))
        learning_rate = float(request.form.get('learning_rate', 5.0))
        content_weight = float(request.form.get('content_weight', 1e4))
        style_weight = float(request.form.get('style_weight', 1e-2))
        
        # Get content image
        content_image_path = None
        if 'content_image' in request.files:
            content_file = request.files['content_image']
            if content_file and content_file.filename and allowed_file(content_file.filename):
                filename = secure_filename(content_file.filename)
                # Add unique prefix to avoid conflicts
                unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
                content_image_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'content', unique_filename
                )
                os.makedirs(os.path.dirname(content_image_path), exist_ok=True)
                content_file.save(content_image_path)
        
        # If no upload, check for default content image
        if not content_image_path:
            default_content = request.form.get('default_content')
            if default_content:
                content_image_path = os.path.join('contentimage', default_content)
                if not os.path.exists(content_image_path):
                    return jsonify({'error': 'Default content image not found'}), 404
            else:
                return jsonify({'error': 'No content image provided'}), 400
        
        # Get style image
        style_image_path = None
        if 'style_image' in request.files:
            style_file = request.files['style_image']
            if style_file and style_file.filename and allowed_file(style_file.filename):
                filename = secure_filename(style_file.filename)
                unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
                style_image_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'style', unique_filename
                )
                os.makedirs(os.path.dirname(style_image_path), exist_ok=True)
                style_file.save(style_image_path)
        
        # If no upload, check for default style image
        if not style_image_path:
            default_style = request.form.get('default_style')
            if default_style:
                style_image_path = os.path.join('style_images', default_style)
                if not os.path.exists(style_image_path):
                    return jsonify({'error': 'Default style image not found'}), 404
            else:
                return jsonify({'error': 'No style image provided'}), 400
        
        # Validate images
        is_valid, error_msg = validate_image(content_image_path)
        if not is_valid:
            return jsonify({'error': f'Invalid content image: {error_msg}'}), 400
        
        is_valid, error_msg = validate_image(style_image_path)
        if not is_valid:
            return jsonify({'error': f'Invalid style image: {error_msg}'}), 400
        
        # Initialize style transfer
        style_transfer = StyleTransfer(content_image_path, style_image_path)
        
        # Generate stylized image
        output_image, _ = style_transfer.generate(
            epochs=epochs,
            steps_per_epoch=5,
            learning_rate=learning_rate,
            content_weight=content_weight,
            style_weight=style_weight
        )
        
        # Save output image
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        output_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        save_image(output_image, output_path)
        
        # Return success with image URL
        image_url = url_for('serve_output', filename=output_filename)
        return jsonify({
            'success': True,
            'image_url': image_url,
            'filename': output_filename
        })
    
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    except Exception as e:
        return jsonify({'error': f'Error generating image: {str(e)}'}), 500


@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve generated output images."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/contentimage/<filename>')
def serve_content_image(filename):
    """Serve default content images."""
    return send_from_directory('contentimage', filename)


@app.route('/style_images/<filename>')
def serve_style_image(filename):
    """Serve default style images."""
    return send_from_directory('style_images', filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size errors."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'content'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'style'), exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

