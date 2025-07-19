// src/components/SongUpload.tsx - FINAL WORKING VERSION
'use client'

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Music, AlertCircle, Loader } from 'lucide-react';
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
    <div className="max-w-2xl mx-auto p-6">
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Song Nerd
        </h1>
        <p className="text-xl text-gray-600">
          Get AI-powered marketing insights for your music
        </p>
      </div>

      {/* Metadata Form */}
      <div className="mb-6 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Song Title
            </label>
            <input
              type="text"
              value={songMetadata.title}
              onChange={(e) => setSongMetadata(prev => ({ ...prev, title: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter song title"
              disabled={uploading}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Artist Name
            </label>
            <input
              type="text"
              value={songMetadata.artist_name}
              onChange={(e) => setSongMetadata(prev => ({ ...prev, artist_name: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter artist name"
              disabled={uploading}
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Genre
          </label>
          <select
            value={songMetadata.genre}
            onChange={(e) => setSongMetadata(prev => ({ ...prev, genre: e.target.value }))}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50 scale-105' 
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
          }
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center">
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
              <Music className="h-16 w-16 text-gray-400 mb-6" />
              {isDragActive ? (
                <p className="text-xl font-medium text-blue-600">
                  Drop your song here! 
                </p>
              ) : (
                <>
                  <p className="text-xl font-medium text-gray-900 mb-2">
                    Drag & drop your song here
                  </p>
                  <p className="text-gray-500 mb-4">
                    or click to browse files
                  </p>
                  <div className="flex items-center space-x-2 text-sm text-gray-400">
                    <span>Supports:</span>
                    <span className="bg-gray-100 px-2 py-1 rounded">MP3</span>
                    <span className="bg-gray-100 px-2 py-1 rounded">WAV</span>
                    <span className="bg-gray-100 px-2 py-1 rounded">M4A</span>
                    <span className="bg-gray-100 px-2 py-1 rounded">FLAC</span>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-400 mr-3" />
            <div>
              <p className="text-sm text-red-800">{error}</p>
              <p className="text-xs text-red-600 mt-1">
                Check browser console for more details
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}