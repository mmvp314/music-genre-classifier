"""
Music Genre Classifier - AI-powered music genre classification
Copyright (C) 2026 Mathilde Pascal

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import os
import io
import uuid
import torch
from datetime import datetime, timedelta

from src.predict import load_model, predict_genre_from_bytes
from config import secret_key

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'raw'}

# In-memory cache for audio files (temporary storage)
# Structure: {uuid: {'audio_bytes': bytes, 'filename': str, 'timestamp': datetime}}
audio_cache = {}

# Load model once at startup
print("="*50)
print("Starting Music Genre Classifier...")
print("="*50)
print("Loading model checkpoint...")
checkpoint_path = os.path.join("outputs", "models", "checkpoint_audioCNN_best.pth")
model = load_model(checkpoint_path)
model.eval()
print("✓ Model loaded successfully!")
print("✓ Server ready at http://127.0.0.1:5000")
print("="*50)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_file(file):
    """
    Validate that uploaded file is actually an audio file
    Uses blacklist approach - rejects known dangerous file types
    Safe for production deployment
    
    Args:
        file: Flask uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check extension first
    if not allowed_file(file.filename):
        return False, f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
    
    # Check filename is not empty after sanitization
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        return False, 'Invalid filename'
    
    # Check file has content
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Back to start
    
    if file_size == 0:
        return False, 'File is empty'
    
    if file_size > 20 * 1024 * 1024:
        return False, 'File too large (max 20MB)'
    
    # Check actual file content (MIME type) if python-magic is available
    try:
        import magic
        
        file.seek(0)
        file_header = file.read(2048)  # Read first 2KB
        file.seek(0)  # Reset to start
        
        mime = magic.from_buffer(file_header, mime=True)
        
        # Blacklist: Reject known dangerous/executable types
        dangerous_mimes = [
            # Executables
            'application/x-executable',
            'application/x-msdos-program', 
            'application/x-msdownload',
            'application/x-dosexec',
            'application/x-elf',
            'application/x-mach-binary',
            
            # Scripts
            'text/x-shellscript',
            'application/x-sh',
            'text/x-python',
            'application/x-python-code',
            
            # Web content (potential XSS)
            'text/html',
            'text/javascript',
            'application/javascript',
            'application/x-javascript',
            
            # Archives (could contain malware)
            'application/zip',
            'application/x-rar',
            'application/x-tar',
            'application/gzip',
            'application/x-7z-compressed',
            
            # Documents (not audio)
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument',
            
            # Images (not audio)
            'image/jpeg',
            'image/png',
            'image/gif',
            'image/bmp'
        ]
        
        if mime in dangerous_mimes:
            return False, f'File type not allowed for security reasons'
        
        # If MIME type is generic (like application/octet-stream) but extension is valid,
        # trust the extension and let librosa validate later
        return True, None
        
    except ImportError:
        # python-magic not installed - fall back to extension check only
        # This is acceptable since librosa will fail safely on non-audio files
        return True, None
        
    except Exception as e:
        # If MIME detection fails, be cautious and reject
        return False, f'Error validating file: {str(e)}'
    
def cleanup_old_cache(max_age_minutes=30):
    """Remove audio files older than max_age_minutes from cache"""
    current_time = datetime.now()
    keys_to_delete = []
    
    for audio_id, data in list(audio_cache.items()):
        if current_time - data['timestamp'] > timedelta(minutes=max_age_minutes):
            keys_to_delete.append(audio_id)
    
    for key in keys_to_delete:
        if key in audio_cache:
            del audio_cache[key]

@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")  
def predict():
    """Handle file upload and prediction"""
    
    # Check if file is present
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio_file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    is_valid, error_msg = validate_audio_file(file)
    if not is_valid:
        return jsonify({'error': error_msg}), 400
        
    try:
        # Read file into memory
        audio_bytes = file.read()
        filename = secure_filename(file.filename)

        if len(audio_bytes) > 20 * 1024 * 1024:
            return jsonify({'error': 'File too large'}), 400
                
        # Clean up old cached files
        cleanup_old_cache()
        
        # Generate unique ID for this audio file
        audio_id = str(uuid.uuid4())
        
        # Store audio in cache for playback
        audio_cache[audio_id] = {
            'audio_bytes': audio_bytes,
            'filename': filename,
            'timestamp': datetime.now()
        }
        
        # Run prediction (this function will be modified to accept bytes)
        prediction_result = predict_genre_from_bytes(model, audio_bytes, filename)
        
        # Add audio_id to result for playback
        prediction_result['audio_id'] = audio_id
        prediction_result['filename'] = filename
        
        return jsonify(prediction_result), 200
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/audio/<audio_id>')
def serve_audio(audio_id):
    """Serve audio file from cache"""
    
    if audio_id not in audio_cache:
        return jsonify({'error': 'Audio file not found or expired'}), 404
    
    audio_data = audio_cache[audio_id]
    
    # Create BytesIO object from cached audio
    audio_io = io.BytesIO(audio_data['audio_bytes'])
    audio_io.seek(0)
    
    # Determine mimetype based on filename extension
    extension = audio_data['filename'].rsplit('.', 1)[1].lower()
    mimetypes = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'flac': 'audio/flac',
        'ogg': 'audio/ogg',
        'raw': 'audio/raw'
    }
    mimetype = mimetypes.get(extension, 'audio/mpeg')
    
    return send_file(
        audio_io,
        mimetype=mimetype,
        as_attachment=False,
        download_name=audio_data['filename']
    )


@app.route('/clear/<audio_id>', methods=['POST'])
def clear_audio(audio_id):
    """Manually remove audio from cache"""
    if audio_id in audio_cache:
        del audio_cache[audio_id]
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Audio not found'}), 404

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)