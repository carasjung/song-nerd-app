# audio_pipeline.py

from audio_processor import AudioFeatureExtractor
from file_handler import AudioFileHandler
from integrated_analyzer import MusicMarketingAnalyzer
import os
import json

class CompleteAudioPipeline:
    def __init__(self):
        self.file_handler = AudioFileHandler()
        self.feature_extractor = AudioFeatureExtractor()
        self.marketing_analyzer = MusicMarketingAnalyzer()
        
        self.marketing_analyzer.load_models()
    
    def process_uploaded_song(self, file_path, metadata=None):
        """Complete pipeline: File → Features → Marketing Analysis"""
        
        results = {
            'success': False,
            'file_info': {},
            'audio_features': {},
            'marketing_analysis': {},
            'errors': []
        }
        
        try:
            # Validate file
            print("🔍 Validating uploaded file...")
            is_valid, validation_errors = self.file_handler.validate_file(file_path)
            
            if not is_valid:
                results['errors'] = validation_errors
                return results
            
            print("File validation passed")
            
            # Convert to standard format
            print("Converting to standard format...")
            processed_path, file_metadata = self.file_handler.convert_to_standard_format(file_path)
            results['file_info'] = file_metadata
            
            print("File conversion completed")
            
            # Extract audio features
            print("🎵 Extracting audio features...")
            audio_features = self.feature_extractor.process_file(processed_path, metadata)
            results['audio_features'] = audio_features
            
            print("Audio feature extraction completed")
            
            # Marketing analysis
            print("Performing marketing analysis...")
            song_metadata = {
                'track_name': metadata.get('track_name', 'Unknown Track'),
                'artist_name': metadata.get('artist_name', 'Unknown Artist'),
                'genre': metadata.get('genre', audio_features.get('genre_clean', 'pop'))
            }
            
            marketing_analysis = self.marketing_analyzer.analyze_song(
                audio_features, song_metadata
            )
            results['marketing_analysis'] = marketing_analysis
            
            print("Marketing analysis completed")
            
            results['success'] = True
            
            # Cleanup temporary files
            self.file_handler.cleanup_temp_files()
            
            return results
            
        except Exception as e:
            results['errors'].append(f"Processing failed: {str(e)}")
            print(f"Pipeline error: {e}")
            return results
    
    def get_quick_summary(self, results):
        """Generate a quick summary for display"""
        if not results['success']:
            return {
                'status': 'failed',
                'message': '; '.join(results['errors'])
            }
        
        analysis = results['marketing_analysis']
        
        return {
            'status': 'success',
            'song_info': {
                'title': analysis['song_info'].get('track_name', 'Unknown'),
                'artist': analysis['song_info'].get('artist_name', 'Unknown'),
                'genre': analysis['audio_features'].get('genre_clean', 'Unknown')
            },
            'top_platform': {
                'name': analysis['platform_recommendations']['top_platform'],
                'score': analysis['platform_recommendations']['top_score'],
                'success_probability': analysis['platform_recommendations']['ranked_recommendations'][0]['success_probability']
            },
            'target_demographic': {
                'age_group': analysis['target_demographics']['primary_age_group'],
                'confidence': analysis['target_demographics']['confidence_scores']['age']
            },
            'sound_profile': analysis['marketing_insights']['positioning']['sound_profile'],
            'top_action': analysis['marketing_insights']['action_items'][0],
            'confidence_level': analysis['analysis_summary']['confidence_level']
        }

# Example use and testing
def test_complete_pipeline():
    """Test the complete audio processing pipeline"""
    
    pipeline = CompleteAudioPipeline()
    
    # Test with sample data
    test_metadata = {
        'track_name': 'Test Song',
        'artist_name': 'Test Artist',
        'genre': 'pop'
    }
    
    print("Testing Complete Audio Processing Pipeline")
    print("=" * 60)
    
    # You'll need to provide an actual audio file for testing
    test_file = 'sample_song.mp3'
    
    if os.path.exists(test_file):
        results = pipeline.process_uploaded_song(test_file, test_metadata)
        
        if results['success']:
            print("\nPipeline success")
            print("=" * 40)
            
            # Show quick summary
            summary = pipeline.get_quick_summary(results)
            print(f"Song: {summary['song_info']['title']} by {summary['song_info']['artist']}")
            print(f"Genre: {summary['song_info']['genre']}")
            print(f"Top Platform: {summary['top_platform']['name']} ({summary['top_platform']['score']:.0f}/100)")
            print(f"Success Probability: {summary['top_platform']['success_probability']:.1%}")
            print(f"Target Audience: {summary['target_demographic']['age_group']}")
            print(f"Sound Profile: {summary['sound_profile']}")
            print(f"Top Action: {summary['top_action']}")
            
            # Save detailed results
            with open('test_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: test_analysis_results.json")
            
        else:
            print("\nPipeline failed")
            print("Errors:", results['errors'])
    
    else:
        print(f"Test file '{test_file}' not found")
        print("Please provide a sample audio file to test the pipeline")

if __name__ == "__main__":
    test_complete_pipeline()