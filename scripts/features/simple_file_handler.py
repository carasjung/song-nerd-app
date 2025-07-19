# simple_file_handler.py

import os
import tempfile
from pathlib import Path
from pydub import AudioSegment
import hashlib

class SimpleAudioFileHandler:
    def __init__(self, max_file_size_mb=50):
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.supported_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        self.temp_dir = tempfile.mkdtemp(prefix='audio_processing_')
    
    def validate_file(self, file_path):
        """Validate uploaded audio file without python-magic"""
        errors = []
        
        # Check if file exists
        if not os.path.exists(file_path):
            errors.append("File does not exist")
            return False, errors
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            errors.append(f"File too large: {file_size/(1024*1024):.1f}MB (max: {self.max_file_size/(1024*1024)}MB)")
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            errors.append(f"Unsupported format: {file_ext} (supported: {', '.join(self.supported_extensions)})")
        
        # Try to load as audio (this validates it's actually an audio file)
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

# Test the simplified file handler
def test_simple_file_handler():
    """Test the simplified file handler"""
    print("Testing Simplified File Handler...")
    
    handler = SimpleAudioFileHandler()
    
    # Look for audio files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    test_file = None
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            test_file = file
            break
    
    if not test_file:
        print("No audio files found for testing")
        return False
    
    print(f"Testing with: {test_file}")
    
    # Test validation
    is_valid, errors = handler.validate_file(test_file)
    if is_valid:
        print("File validation passed")
        
        # Test conversion
        try:
            converted_path, metadata = handler.convert_to_standard_format(test_file)
            print(f"File converted successfully")
            print(f"   Output: {converted_path}")
            print(f"   Duration: {metadata['duration_seconds']:.1f}s")
            print(f"   Sample rate: {metadata['sample_rate']}Hz")
            print(f"   Size: {metadata['file_size_mb']:.1f}MB")
            
            # Cleanup
            handler.cleanup_temp_files()
            print("Cleanup completed")
            return True
            
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False
    else:
        print(f"File validation failed: {errors}")
        return False

if __name__ == "__main__":
    test_simple_file_handler()