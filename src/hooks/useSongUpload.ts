// src/hooks/useSongUpload.ts
import { useState } from 'react';
import { songAPI } from '@/services/api';
import { supabase } from '@/lib/supabase';

interface SongMetadata {
  title: string;
  artist_name: string;
  genre: string;
}

interface UploadState {
  uploading: boolean;
  progress: number;
  step: string;
  error: string | null;
}

export const useSongUpload = () => {
  const [state, setState] = useState<UploadState>({
    uploading: false,
    progress: 0,
    step: '',
    error: null
  });

  const uploadSong = async (file: File, metadata: SongMetadata) => {
    setState({
      uploading: true,
      progress: 0,
      step: 'Preparing upload...',
      error: null
    });

    try {
      // Upload file to Supabase Storage
      setState(prev => ({ ...prev, step: 'Uploading file...', progress: 20 }));
      const { url: fileUrl } = await songAPI.uploadFile(file, 'songs');

      // Create song record
      setState(prev => ({ ...prev, step: 'Creating song record...', progress: 40 }));
      const songData = {
        title: metadata.title || file.name.replace(/\.[^/.]+$/, ""),
        artist_name: metadata.artist_name || 'Unknown Artist',
        genre: metadata.genre,
        file_path: fileUrl,
        file_size: file.size,
        duration: null,
        processing_status: 'pending',
        user_id: null,
      };

      const { data: songRecord, error: dbError } = await supabase
        .from('songs')
        .insert(songData)
        .select()
        .single();

      if (dbError) throw new Error(`Database error: ${dbError.message}`);

      // Trigger analysis
      setState(prev => ({ ...prev, step: 'Starting AI analysis...', progress: 60 }));
      await songAPI.triggerAnalysis(songRecord.id, fileUrl, metadata);

      // Update status
      setState(prev => ({ ...prev, step: 'Analysis in progress...', progress: 80 }));
      await songAPI.updateSongStatus(songRecord.id, 'processing');

      setState(prev => ({ ...prev, step: 'Complete!', progress: 100 }));

      // Reset after delay
      setTimeout(() => {
        setState({
          uploading: false,
          progress: 0,
          step: '',
          error: null
        });
      }, 1000);

      return songRecord;

    } catch (error: any) {
      setState(prev => ({
        ...prev,
        uploading: false,
        error: error.message || 'Upload failed',
        step: ''
      }));
      throw error;
    }
  };

  const checkBackendHealth = async () => {
    try {
      return await songAPI.checkBackendHealth();
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  };

  return {
    ...state,
    uploadSong,
    checkBackendHealth
  };
};