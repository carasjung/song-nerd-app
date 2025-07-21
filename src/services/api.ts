// src/services/api.ts
import { supabase } from '@/lib/supabase';

// Get the API URL from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://song-nerd.up.railway.app';

// Ensure we're using HTTPS
const secureApiUrl = API_BASE_URL.replace('http://', 'https://');

// Debug: Log the API URL
console.log('🔧 API_BASE_URL:', API_BASE_URL);
console.log('🔧 process.env.NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);

export const songAPI = {
  // Get user's songs
  getUserSongs: async (userId?: string) => {
    let query = supabase.from('songs').select('*');
    
    if (userId) {
      query = query.eq('user_id', userId);
    }
    
    const { data, error } = await query.order('created_at', { ascending: false });
    
    if (error) throw error;
    return data;
  },

  // Get song with analysis and insights
  getSongWithAnalysis: async (songId: string) => {
    const { data: song, error: songError } = await supabase
      .from('songs')
      .select('*')
      .eq('id', songId)
      .single();
    if (songError) throw songError;

    const { data: analysis } = await supabase
      .from('analysis')
      .select('*')
      .eq('song_id', songId)
      .single();

    const { data: insights } = await supabase
      .from('marketing_insights')
      .select('*')
      .eq('song_id', songId)
      .single();

    return { song, analysis, insights };
  },

  // Update song status
  updateSongStatus: async (songId: string, status: string) => {
    const { data, error } = await supabase
      .from('songs')
      .update({ processing_status: status })
      .eq('id', songId)
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  // Upload file to Supabase Storage - FIXED VERSION
  uploadFile: async (file: File, bucket: string = 'songs') => {
    try {
      console.log('📁 Starting file upload to Supabase Storage...');
      
      // Generate unique filename
      const fileExt = file.name.split('.').pop();
      const fileName = `${Date.now()}-${Math.random().toString(36).substring(2)}.${fileExt}`;
      
      console.log(`📁 Uploading as: ${fileName}`);
      
      // Upload to Supabase Storage
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from(bucket)
        .upload(fileName, file, {
          cacheControl: '3600',
          upsert: false
        });
      
      if (uploadError) {
        console.error('❌ Storage upload error:', uploadError);
        throw new Error(`Storage upload failed: ${uploadError.message}`);
      }
      
      console.log('✅ Upload successful:', uploadData);
      
      // Get public URL
      const { data: urlData } = supabase.storage
        .from(bucket)
        .getPublicUrl(fileName);
      
      if (!urlData?.publicUrl) {
        throw new Error('Failed to get public URL');
      }
      
      console.log('✅ Public URL generated:', urlData.publicUrl);
      
      return { 
        path: uploadData.path, 
        url: urlData.publicUrl 
      };
      
    } catch (error: any) {
      console.error('💥 Upload file error:', error);
      throw new Error(`File upload failed: ${error.message}`);
    }
  },

  // Trigger analysis via Railway backend (from URL)
  triggerAnalysis: async (songId: string, fileUrl: string, metadata: any) => {
    try {
            console.log('🚀 Triggering analysis with:', { songId, fileUrl, metadata });
      console.log('🌐 API URL:', `${secureApiUrl}/api/songs/analyze`);
      console.log('🔧 API_BASE_URL type:', typeof API_BASE_URL);
      console.log('🔧 API_BASE_URL length:', API_BASE_URL?.length);
      console.log('🔧 Secure API URL:', secureApiUrl);
      
      if (!secureApiUrl) {
        throw new Error('Secure API URL is undefined or empty');
      }
      
      const fullUrl = `${secureApiUrl}/api/songs/analyze`;
      console.log('🔧 Full URL:', fullUrl);
      console.log('🔧 URL validation:', {
        hasProtocol: fullUrl.startsWith('https'),
        hasHost: fullUrl.includes('song-nerd.up.railway.app'),
        urlLength: fullUrl.length
      });
      
      let response;
      try {
        response = await fetch(fullUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        song_id: songId,
        file_url: fileUrl,
        metadata: metadata
      }),
    });
      } catch (fetchError: any) {
        console.error('💥 Fetch error:', fetchError);
        throw new Error(`Network error: ${fetchError.message}`);
      }

      console.log('📡 Response status:', response.status);
      console.log('📡 Response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Analysis request failed:', errorText);
        throw new Error(`Analysis request failed: ${response.status} ${response.statusText} - ${errorText}`);
    }

      const result = await response.json();
      console.log('✅ Analysis triggered successfully:', result);
      return result;
    } catch (error) {
      console.error('💥 Analysis trigger error:', error);
      throw error;
    }
  },

  // Upload and analyze via Railway backend (direct file upload)
  uploadAndAnalyze: async (file: File, songId: string, metadata: any) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('song_id', songId);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch(`${secureApiUrl}/api/songs/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload and analysis failed: ${response.statusText}`);
    }

    return response.json();
  },

  // Check backend health
  checkBackendHealth: async () => {
    try {
      const response = await fetch(`${secureApiUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }
};