// src/components/ProcessingStatus.tsx
'use client'

import React, { useState, useEffect } from 'react';
import { Loader, CheckCircle, AlertCircle, Music } from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface ProcessingStatusProps {
  songId: string;
  onComplete: () => void;
}

export default function ProcessingStatus({ songId, onComplete }: ProcessingStatusProps) {
  const [status, setStatus] = useState('pending');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        // Check song status from Supabase
        const { data: song, error } = await supabase
          .from('songs')
          .select('processing_status')
          .eq('id', songId)
          .single();

        if (error) {
          console.error('Error fetching song status:', error);
          setStatus('failed');
          return;
        }
        
        const currentStatus = song.processing_status;
        setStatus(currentStatus);

        // Simulate progress for better UX
        if (currentStatus === 'pending') setProgress(10);
        else if (currentStatus === 'processing') setProgress(60);
        else if (currentStatus === 'completed') {
          setProgress(100);
          setTimeout(() => onComplete(), 1000);
        } else if (currentStatus === 'failed') {
          setProgress(0);
        }

      } catch (error) {
        console.error('Error checking status:', error);
        setStatus('failed');
      }
    };

    // Set up real-time subscription for status updates
    const subscription = supabase
      .channel(`song-${songId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'songs',
          filter: `id=eq.${songId}`,
        },
        (payload) => {
          const newStatus = payload.new.processing_status;
          setStatus(newStatus);
          
          if (newStatus === 'pending') setProgress(10);
          else if (newStatus === 'processing') setProgress(60);
          else if (newStatus === 'completed') {
            setProgress(100);
            setTimeout(() => onComplete(), 1000);
          } else if (newStatus === 'failed') {
            setProgress(0);
          }
        }
      )
      .subscribe();

    // Initial check
    checkStatus();

    // Fallback polling in case real-time doesn't work
    const interval = setInterval(checkStatus, 5000);

    return () => {
      subscription.unsubscribe();
      clearInterval(interval);
    };
  }, [songId, onComplete]);

  const getStatusIcon = () => {
    switch (status) {
      case 'pending':
        return <Music className="h-8 w-8 text-blue-500 animate-pulse" />;
      case 'processing':
        return <Loader className="h-8 w-8 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-8 w-8 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-8 w-8 text-red-500" />;
      default:
        return <Loader className="h-8 w-8 text-gray-500" />;
    }
  };

  const getStatusMessage = () => {
    switch (status) {
      case 'pending':
        return 'Your song is in the queue...';
      case 'processing':
        return 'Analyzing your music with AI...';
      case 'completed':
        return 'Analysis complete!';
      case 'failed':
        return 'Analysis failed. Please try again.';
      default:
        return 'Processing...';
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="flex justify-center mb-6">
          {getStatusIcon()}
        </div>
        
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Processing Your Song
        </h2>
        
        <p className="text-lg text-gray-600 mb-6">
          {getStatusMessage()}
        </p>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
          <div 
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="text-sm text-gray-500">
          <p>This usually takes 30-60 seconds</p>
          <p className="mt-2">We're extracting audio features and generating marketing insights...</p>
        </div>
      </div>
    </div>
  );
}