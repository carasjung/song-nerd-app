// src/app/page.tsx
'use client'

import React, { useState } from 'react';
import Layout from '@/components/Layout';
import SongUpload from '@/components/SongUpload';
import ProcessingStatus from '@/components/ProcessingStatus';
import AnalysisResults from '@/components/AnalysisResults';

type ViewType = 'upload' | 'processing' | 'results';

interface SongData {
  id: string;
  title: string;
  artist_name: string;
  genre: string;
  processing_status: string;
  file_url?: string;
}

export default function Home() {
  const [view, setView] = useState<ViewType>('upload');
  const [currentSong, setCurrentSong] = useState<SongData | null>(null);

  const handleUploadSuccess = (song: SongData) => {
    setCurrentSong(song);
    setView('processing');
  };

  const handleProcessingComplete = () => {
    setView('results');
  };

  const handleBackToUpload = () => {
    setCurrentSong(null);
    setView('upload');
  };

  return (
    <Layout currentPage="upload">
      {view === 'upload' && (
        <SongUpload onUploadSuccess={handleUploadSuccess} />
      )}
      
      {view === 'processing' && currentSong && (
        <ProcessingStatus 
          songId={currentSong.id} 
          onComplete={handleProcessingComplete} 
        />
      )}
      
      {view === 'results' && currentSong && (
        <AnalysisResults 
          songId={currentSong.id} 
          onBackToUpload={handleBackToUpload} 
        />
      )}
    </Layout>
  );
}