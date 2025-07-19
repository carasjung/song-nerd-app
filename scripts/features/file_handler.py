# file_handler.py

import os
import tempfile
from pathlib import Path
import magic  # python-magic for file type detection
from pydub import AudioSegment
import hashlib

class AudioFileHandler:
    def __init__(self, max_file_size_mb=50, supported_formats=None):
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.supported_formats = supported_formats or [
            'audio/mpeg',      # MP3
            'audio/wav',       # WAV
            'audio/x-wav',     # WAV alternative
            'audio/mp4',       # M4A
            'audio/x-m4a',     # M4A alternative
            'audio/flac',      # FLAC
            'audio/ogg'        # OGG
        ]
        self.temp_dir = tempfile.mkdtemp(prefix='audio_processing_')
    
    def validate_file(self, file_path):
        """Validate uploaded audio file"""
        errors = []
        
        # Check if file exists
        if not os.path.exists(file_path):
            errors.append("File does not exist")
            return False, errors
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            errors.append(f"File too large: {file_size/(1024*1024):.1f}MB (max: {self.max_file_size/(1024*1024)}MB)")
        
        # Check file type
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type not in self.supported_formats:
                errors.append(f"Unsupported format: {mime_type}")
        except Exception as e:
            errors.append(f"Could not determine file type: {e}")
        
        # Try to load as audio
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Check duration (5 seconds minimum, 10 minutes maximum)
            duration_seconds = len(audio) / 1000
            if duration_seconds < 5:
                errors.append(f"Track too short: {duration_seconds:.1f}s (minimum: 5s)")
            elif duration_seconds > 600:  # 10 minutes
                errors.append(f"Track too long: {duration_seconds/60:.1f}min (maximum: 10min)")
            
        except Exception as e:
            errors.append(f"Invalid audio file: {e}")
        
        return len(errors) == 0, errors
    
    def convert_to_standard_format(self, input_path, output_format='wav'):
        """Convert audio file to standard format for processing"""
        try:
            audio = AudioSegment.from_file(input_path)
            
            # Normalize audio settings
            audio = audio.set_frame_rate(22050)  # Standard sample rate
            audio = audio.set_channels(1)       # Convert to mono
            
            # Generate output filename
            file_hash = hashlib.md5(open(input_path, 'rb').read()).hexdigest()[:8]
            output_filename = f"processed_{file_hash}.{output_format}"
            output_path = os.path.join(self.temp_dir, output_filename)
            
            # Export in standard format
            audio.export(output_path, format=output_format)
            
            return output_path, {
                'original_format': Path(input_path).suffix,
                'duration_seconds': len(audio) / 1000,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'file_size_mb': os.path.getsize(output_path) / (1024*1024)
            }
            
        except Exception as e:
            raise Exception(f"Conversion failed: {e}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")

# Example usage
def test_file_handler():
    handler = AudioFileHandler()
    
    # Test validation
    is_valid, errors = handler.validate_file('sample_song.mp3')
    if is_valid:
        print("File validation passed")
        
        # Test conversion
        try:
            converted_path, metadata = handler.convert_to_standard_format('sample_song.mp3')
            print(f"File converted: {converted_path}")
            print(f"   Metadata: {metadata}")
            return converted_path
        except Exception as e:
            print(f"Conversion failed: {e}")
    else:
        print(f"File validation failed: {errors}")
    
    return None

if __name__ == "__main__":
    test_file_handler()