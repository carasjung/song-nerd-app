// src/components/AnalysisResults.tsx
'use client'

import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { 
  TrendingUp, Users, Target, Music2, Copy, ExternalLink, 
  Download, Share, Star, Headphones 
} from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface AnalysisResultsProps {
  songId: string;
  onBackToUpload: () => void;
}

export default function AnalysisResults({ songId, onBackToUpload }: AnalysisResultsProps) {
  const [analysis, setAnalysis] = useState<any>(null);
  const [marketingInsights, setMarketingInsights] = useState<any>(null);
  const [song, setSong] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        // Fetch song details
        const { data: songData, error: songError } = await supabase
          .from('songs')
          .select('*')
          .eq('id', songId)
          .single();

        if (songError) throw songError;
        setSong(songData);

        // Fetch audio analysis
        const { data: analysisData, error: analysisError } = await supabase
          .from('analysis')
          .select('*')
          .eq('song_id', songId)
          .single();

        if (analysisError && analysisError.code !== 'PGRST116') {
          // PGRST116 is "not found" - that's ok, analysis might not exist yet
          console.warn('Analysis not found:', analysisError);
        } else if (analysisData) {
          setAnalysis(analysisData);
        }

        // Fetch marketing insights
        const { data: insightsData, error: insightsError } = await supabase
          .from('marketing_insights')
          .select('*')
          .eq('song_id', songId)
          .single();

        if (insightsError && insightsError.code !== 'PGRST116') {
          console.warn('Marketing insights not found:', insightsError);
        } else if (insightsData) {
          setMarketingInsights(insightsData);
        }

        // If no analysis data exists yet, create mock data for demo
        if (!analysisData && !insightsData) {
          createMockData(songData);
        }

      } catch (err: any) {
        console.error('Error fetching analysis:', err);
        setError(err.message || 'Failed to fetch analysis');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [songId]);

  const createMockData = (songData: any) => {
    // Create mock analysis data for demo purposes
    const mockAnalysis = {
      danceability: 0.7 + Math.random() * 0.3,
      energy: 0.6 + Math.random() * 0.4,
      valence: 0.5 + Math.random() * 0.5,
      acousticness: Math.random() * 0.8,
      tempo: 120 + Math.random() * 60,
      key: Math.floor(Math.random() * 12),
      mode: Math.round(Math.random()),
      audio_appeal: 70 + Math.random() * 30,
    };

    const mockInsights = {
      primary_age_group: '18-24',
      age_confidence: 0.85,
      primary_region: 'North America',
      region_confidence: 0.78,
      top_platform: 'spotify',
      platform_scores: {
        spotify: 85 + Math.random() * 15,
        tiktok: 70 + Math.random() * 20,
        instagram: 75 + Math.random() * 20,
        youtube: 80 + Math.random() * 15,
      },
      similar_artists: [
        { artist_name: 'Taylor Swift', genre: songData.genre, similarity_score: 0.8 },
        { artist_name: 'Ed Sheeran', genre: songData.genre, similarity_score: 0.75 },
        { artist_name: 'Billie Eilish', genre: songData.genre, similarity_score: 0.7 },
      ],
      action_items: [
        'Focus marketing on 18-24 age group',
        'Prioritize Spotify playlist submissions',
        'Create TikTok-friendly short clips',
        'Develop Instagram story content',
        'Consider collaboration opportunities',
      ],
      sound_profile: 'Modern pop with electronic influences',
      competitive_advantage: 'Strong melodic hooks with contemporary production',
      overall_confidence: 0.82,
    };

    setAnalysis(mockAnalysis);
    setMarketingInsights(mockInsights);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-800 mb-4">{error}</p>
          <button
            onClick={onBackToUpload}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Try Another Song
          </button>
        </div>
      </div>
    );
  }

  if (!analysis || !marketingInsights) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <p className="text-yellow-800 mb-4">Analysis is still being processed...</p>
          <button
            onClick={onBackToUpload}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Upload Another Song
          </button>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const platformData = Object.entries(marketingInsights.platform_scores).map(([platform, score]) => ({
    name: platform.charAt(0).toUpperCase() + platform.slice(1),
    score: Math.round(score as number),
    success_probability: Math.round((score as number) * 0.9), // Approximate
  }));

  const audioFeatureData = [
    { name: 'Danceability', value: Math.round(analysis.danceability * 100), color: '#3B82F6' },
    { name: 'Energy', value: Math.round(analysis.energy * 100), color: '#10B981' },
    { name: 'Valence', value: Math.round(analysis.valence * 100), color: '#F59E0B' },
    { name: 'Acousticness', value: Math.round(analysis.acousticness * 100), color: '#EF4444' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">Song Nerd</h1>
            </div>
            <button
              onClick={onBackToUpload}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Analyze Another Song
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6 space-y-8">
        {/* Hero Section */}
        <div className="text-center py-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Marketing Analysis Complete!
          </h1>
          <p className="text-xl text-gray-600 mb-4">
            Analysis for "{song?.title}" by {song?.artist_name}
          </p>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <Target className="h-10 w-10 text-blue-500 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Target Audience</h3>
                <p className="text-2xl font-bold text-blue-600">
                  {marketingInsights.primary_age_group}
                </p>
                <p className="text-sm text-gray-500">
                  {Math.round(marketingInsights.age_confidence * 100)}% confidence
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <TrendingUp className="h-10 w-10 text-green-500 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Top Platform</h3>
                <p className="text-2xl font-bold text-green-600 capitalize">
                  {marketingInsights.top_platform}
                </p>
                <p className="text-sm text-gray-500">
                  {Math.round(marketingInsights.platform_scores[marketingInsights.top_platform])}/100 score
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <Music2 className="h-10 w-10 text-purple-500 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Audio Appeal</h3>
                <p className="text-2xl font-bold text-purple-600">
                  {Math.round(analysis.audio_appeal)}/100
                </p>
                <p className="text-sm text-gray-500">Production quality</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <Headphones className="h-10 w-10 text-orange-500 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Success Rate</h3>
                <p className="text-2xl font-bold text-orange-600">
                  {Math.round(marketingInsights.overall_confidence * 100)}%
                </p>
                <p className="text-sm text-gray-500">Predicted success</p>
              </div>
            </div>
          </div>
        </div>

        {/* Platform Recommendations Chart */}
        <div className="bg-white p-8 rounded-xl shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Platform Performance Predictions</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={platformData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'score' ? `${value}/100` : `${value}%`,
                  name === 'score' ? 'Performance Score' : 'Success Probability'
                ]}
              />
              <Bar dataKey="score" fill="#3B82F6" name="score" />
              <Bar dataKey="success_probability" fill="#10B981" name="success_probability" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Audio Features Visualization */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Audio Characteristics</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={audioFeatureData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {audioFeatureData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Track Details</h2>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Tempo:</span>
                <span className="text-gray-900">{Math.round(analysis.tempo)} BPM</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Key:</span>
                <span className="text-gray-900">Key {analysis.key}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Mode:</span>
                <span className="text-gray-900">{analysis.mode === 1 ? 'Major' : 'Minor'}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Genre:</span>
                <span className="text-gray-900 capitalize">{song?.genre}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Marketing Recommendations */}
        <div className="bg-white p-8 rounded-xl shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Marketing Action Plan</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommended Actions</h3>
              <div className="space-y-3">
                {marketingInsights.action_items.map((action: string, index: number) => (
                  <div key={index} className="flex items-start">
                    <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-semibold mr-3 mt-0.5">
                      {index + 1}
                    </div>
                    <p className="text-gray-700">{action}</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Similar Artists</h3>
              <div className="space-y-4">
                {marketingInsights.similar_artists.map((artist: any, index: number) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold">{artist.artist_name}</h4>
                      <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {Math.round(artist.similarity_score * 100)}% match
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 capitalize">{artist.genre}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Summary */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">Summary</h2>
          <p className="text-lg opacity-90 mb-4">
            {marketingInsights.sound_profile} - {marketingInsights.competitive_advantage}
          </p>
          <p className="text-base opacity-80 mb-6">
            Target your {marketingInsights.primary_age_group} audience primarily on {marketingInsights.top_platform} 
            for maximum impact.
          </p>
          <div className="flex flex-wrap gap-4">
            <button className="bg-white text-blue-600 px-6 py-2 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              <Download className="h-4 w-4 inline mr-2" />
              Export Report
            </button>
            <button className="bg-white/20 text-white px-6 py-2 rounded-lg font-semibold hover:bg-white/30 transition-colors">
              <Share className="h-4 w-4 inline mr-2" />
              Share Results
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}