// src/components/SongUpload.tsx - FINAL WORKING VERSION
'use client'

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Music, AlertCircle, Loader } from 'lucide-react';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import { songAPI } from '@/services/api';

// Debug: Check if environment variables are loaded
console.log('Environment Debug:', {
  supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
  hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  anonKeyStart: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.substring(0, 20) + '...'
});

interface UploadResponse {
  id: string;
  title: string;
  artist_name: string;
  genre: string;
  processing_status: string;
}

interface SongUploadProps {
  onUploadSuccess: (song: UploadResponse) => void;
}

export default function SongUpload({ onUploadSuccess }: SongUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [songMetadata, setSongMetadata] = useState({
    title: '',
    artist_name: '',
    genre: 'pop'
  });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    console.log('🎵 Starting full upload and analysis process...');
    setUploading(true);
    setError(null);
    setUploadProgress(0);
    setCurrentStep('Preparing upload...');

    try {
      // Upload file to Supabase Storage
      setCurrentStep('Uploading file...');
      setUploadProgress(20);
      
      const { url: fileUrl, path: filePath } = await songAPI.uploadFile(file, 'songs');
      console.log('File uploaded to:', fileUrl);

      // Create song record in database
      setCurrentStep('Creating song record...');
      setUploadProgress(40);

      const songData = {
        title: songMetadata.title || file.name.replace(/\.[^/.]+$/, ""),
        artist_name: songMetadata.artist_name || 'Unknown Artist',
        genre: songMetadata.genre,
        file_path: fileUrl,
        file_size: file.size,
        duration: null,
        processing_status: 'pending',
        user_id: null,
      };

      console.log('Creating song record with:', songData);

      const { data: newSongRecord, error: dbError } = await supabase
        .from('songs')
        .insert([songData])
        .select()
        .single();

      if (dbError) {
        console.error('Database error:', dbError);
        throw new Error(`Database error: ${dbError.message}`);
      }

      console.log('Song record created:', newSongRecord);

      // Trigger AI analysis via Railway backend
      setCurrentStep('Starting AI analysis...');
      setUploadProgress(60);

      const metadata = {
        title: newSongRecord.title,
        artist: newSongRecord.artist_name,
        genre: newSongRecord.genre,
        file_size: newSongRecord.file_size
      };

      console.log('Triggering analysis via Railway backend...');
      const analysisResponse = await songAPI.triggerAnalysis(
        newSongRecord.id,
        fileUrl,
        metadata
      );

      console.log('Analysis started:', analysisResponse);

      // Update song status to processing
      setCurrentStep('Analysis in progress...');
      setUploadProgress(80);

      await songAPI.updateSongStatus(newSongRecord.id, 'processing');

      setCurrentStep('Complete!');
      setUploadProgress(100);

      // Success! Pass the song record to parent component
      setTimeout(() => {
        onUploadSuccess(newSongRecord);
      }, 500);

    } catch (err: any) {
      console.error('Error:', err);
      setError(err.message || 'Upload failed');
      setCurrentStep('');
    } finally {
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setCurrentStep('');
      }, 1000);
    }
  }, [songMetadata, onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.flac']
    },
    maxFiles: 1,
    disabled: uploading
  });

  return (
    <div className="upload-container">
      <div className="upload-subtitle">
        <p>
          Get AI-powered marketing insights for your music
        </p>
      </div>

      {/* Metadata Form */}
      <div className="upload-form">
        <div className="upload-form-grid">
          <div className="upload-form-group">
            <label className="upload-label">
              Song Title
            </label>
            <input
              type="text"
              value={songMetadata.title}
              onChange={(e) => setSongMetadata(prev => ({ ...prev, title: e.target.value }))}
              className="upload-input"
              placeholder="Enter song title"
              disabled={uploading}
            />
          </div>
          
          <div className="upload-form-group">
            <label className="upload-label">
              Artist Name
            </label>
            <input
              type="text"
              value={songMetadata.artist_name}
              onChange={(e) => setSongMetadata(prev => ({ ...prev, artist_name: e.target.value }))}
              className="upload-input"
              placeholder="Enter artist name"
              disabled={uploading}
            />
          </div>
        </div>
        
        <div className="upload-form-group">
          <label className="upload-label">
            Genre
          </label>
          <select
            value={songMetadata.genre}
            onChange={(e) => setSongMetadata(prev => ({ ...prev, genre: e.target.value }))}
            className="upload-select genre-select"
            style={{
              backgroundColor: 'white',
              border: '1px solid #D1D5DB',
              color: '#111827'
            }}
            disabled={uploading}
          >
            <option value="pop">Pop</option>
            <option value="rock">Rock</option>
            <option value="hip hop">Hip Hop</option>
            <option value="electronic">Electronic</option>
            <option value="country">Country</option>
            <option value="r&b">R&B</option>
            <option value="indie">Indie</option>
            <option value="folk">Folk</option>
          </select>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div
        {...getRootProps()}
        className={`upload-dropzone ${isDragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="upload-dropzone-content">
          {uploading ? (
            <>
              <Loader className="h-16 w-16 text-blue-500 animate-spin mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                {currentStep}
              </p>
              <div className="w-full max-w-xs bg-gray-200 rounded-full h-2 mb-4">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-500">{uploadProgress}% complete</p>
            </>
          ) : (
            <>
              {isDragActive ? (
                  <p className="upload-title text-blue-600">
                  Drop your song here! 
                </p>
              ) : (
                <>
                    <p className="upload-title">
                    Drag & drop your song here
                  </p>
                    <p className="upload-subtitle-text">
                    or click to browse files
                  </p>
                    <div className="upload-file-types">
                    <span>Supports:</span>
                      <span className="upload-file-type-tag">MP3</span>
                      <span className="upload-file-type-tag">WAV</span>
                      <span className="upload-file-type-tag">M4A</span>
                      <span className="upload-file-type-tag">FLAC</span>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="upload-error">
          <div className="upload-error-content">
            <AlertCircle className="upload-error-icon" />
            <div>
              <p className="upload-error-text">{error}</p>
              <p className="upload-error-details">
                Check browser console for more details
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}