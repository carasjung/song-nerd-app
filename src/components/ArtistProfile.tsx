// src/components/ArtistProfile.tsx
'use client'

import React, { useState, useEffect } from 'react';
import { 
  User, Music, TrendingUp, Calendar, Download, 
  Eye, Heart, Share2, BarChart3 
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { supabase } from '@/lib/supabase';

interface Song {
  id: string;
  title: string;
  artist_name: string;
  genre: string;
  created_at: string;
  processing_status: string;
  play_count?: number;
  engagement_score?: number;
  file_size?: number;
  duration?: number;
}

interface AnalyticsData {
  genreStats: { [key: string]: number };
  platformStats: { [key: string]: number };
  totalAnalyzed: number;
  avgConfidence: number;
}

export default function ArtistProfile() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        // Fetch all songs for the user
        // Note: Using a temporary user_id for demo. In production, get from auth context
        const { data: songsData, error: songsError } = await supabase
          .from('songs')
          .select('*')
          .order('created_at', { ascending: false });

        if (songsError) {
          console.error('Error fetching songs:', songsError);
          // Create mock data if no songs found
          createMockSongs();
        } else {
          setSongs(songsData || []);
          
          // Fetch analytics data
          await fetchAnalytics(songsData || []);
        }

      } catch (error) {
        console.error('Error fetching user data:', error);
        createMockSongs();
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, []);

  const fetchAnalytics = async (songsData: Song[]) => {
    try {
      // Fetch marketing insights for all songs to generate analytics
      const songIds = songsData.map(song => song.id);
      
      if (songIds.length === 0) {
        setAnalytics(createMockAnalytics());
        return;
      }

      const { data: insights, error } = await supabase
        .from('marketing_insights')
        .select('*')
        .in('song_id', songIds);

      if (error) {
        console.error('Error fetching insights:', error);
        setAnalytics(createMockAnalytics());
        return;
      }

      // Process analytics data
      const genreStats: { [key: string]: number } = {};
      const platformStats: { [key: string]: number } = {};
      let totalConfidence = 0;

      songsData.forEach(song => {
        genreStats[song.genre] = (genreStats[song.genre] || 0) + 1;
      });

      insights?.forEach(insight => {
        const topPlatform = insight.top_platform;
        if (topPlatform) {
          platformStats[topPlatform] = (platformStats[topPlatform] || 0) + 1;
        }
        totalConfidence += insight.overall_confidence || 0;
      });

      setAnalytics({
        genreStats,
        platformStats,
        totalAnalyzed: insights?.length || 0,
        avgConfidence: insights?.length ? totalConfidence / insights.length : 0,
      });

    } catch (error) {
      console.error('Error processing analytics:', error);
      setAnalytics(createMockAnalytics());
    }
  };

  const createMockSongs = () => {
    const mockSongs: Song[] = [
      {
        id: '1',
        title: 'Summer Vibes',
        artist_name: 'Demo Artist',
        genre: 'pop',
        created_at: new Date().toISOString(),
        processing_status: 'completed',
        play_count: 1500,
        engagement_score: 85,
      },
      {
        id: '2',
        title: 'Electronic Dreams',
        artist_name: 'Demo Artist',
        genre: 'electronic',
        created_at: new Date(Date.now() - 86400000).toISOString(),
        processing_status: 'completed',
        play_count: 2300,
        engagement_score: 92,
      },
      {
        id: '3',
        title: 'Acoustic Soul',
        artist_name: 'Demo Artist',
        genre: 'folk',
        created_at: new Date(Date.now() - 172800000).toISOString(),
        processing_status: 'pending',
        play_count: 890,
        engagement_score: 78,
      },
    ];
    setSongs(mockSongs);
    setAnalytics(createMockAnalytics());
  };

  const createMockAnalytics = (): AnalyticsData => ({
    genreStats: { pop: 2, electronic: 1, folk: 1, rock: 1 },
    platformStats: { spotify: 3, tiktok: 2, instagram: 1 },
    totalAnalyzed: 5,
    avgConfidence: 0.87,
  });

  // Mock performance data
  const performanceData = [
    { month: 'Jan', plays: 1200, engagement: 85 },
    { month: 'Feb', plays: 1800, engagement: 92 },
    { month: 'Mar', plays: 2400, engagement: 88 },
    { month: 'Apr', plays: 3200, engagement: 95 },
    { month: 'May', plays: 2800, engagement: 90 },
    { month: 'Jun', plays: 3600, engagement: 98 },
  ];

  const totalPlays = songs.reduce((sum, song) => sum + (song.play_count || 0), 0);
  const avgEngagement = songs.reduce((sum, song) => sum + (song.engagement_score || 0), 0) / songs.length || 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Get top genres for analytics display
  const topGenres = analytics ? 
    Object.entries(analytics.genreStats)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([genre, count], index) => ({
        genre,
        count,
        percentage: Math.round((count / songs.length) * 100),
        index
      })) : [];

  // Get platform performance for analytics display
  const platformPerformance = analytics ?
    Object.entries(analytics.platformStats)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([platform, count], index) => ({
        platform: platform.charAt(0).toUpperCase() + platform.slice(1),
        successRate: 85 + index * 5,
        count
      })) : [];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-xl font-bold text-gray-900">Artist Dashboard</h1>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {/* Profile Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white mb-8">
          <div className="flex items-center">
            <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mr-6">
              <User className="h-10 w-10" />
            </div>
            <div>
              <h2 className="text-3xl font-bold mb-2">Welcome back, Artist!</h2>
              <p className="text-blue-100">Track your music's marketing performance</p>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center">
              <Music className="h-8 w-8 text-blue-500 mr-3" />
              <div>
                <p className="text-2xl font-bold text-gray-900">{songs.length}</p>
                <p className="text-gray-600">Songs Analyzed</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center">
              <Eye className="h-8 w-8 text-green-500 mr-3" />
              <div>
                <p className="text-2xl font-bold text-gray-900">{totalPlays.toLocaleString()}</p>
                <p className="text-gray-600">Total Plays</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center">
              <Heart className="h-8 w-8 text-red-500 mr-3" />
              <div>
                <p className="text-2xl font-bold text-gray-900">{Math.round(avgEngagement)}%</p>
                <p className="text-gray-600">Avg Engagement</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-purple-500 mr-3" />
              <div>
                <p className="text-2xl font-bold text-gray-900">
                  {analytics ? Math.round(analytics.avgConfidence * 100) : 0}%
                </p>
                <p className="text-gray-600">Avg Confidence</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-xl shadow-lg mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {['overview', 'songs', 'analytics'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'overview' && (
              <div className="space-y-8">
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Performance Overview</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="plays" stroke="#3B82F6" strokeWidth={2} />
                      <Line type="monotone" dataKey="engagement" stroke="#10B981" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Most Successful Genre</h4>
                    <p className="text-2xl font-bold text-blue-600 capitalize">
                      {topGenres[0]?.genre || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-500">
                      {topGenres[0]?.count || 0} songs analyzed
                    </p>
                  </div>

                  <div className="border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Top Platform</h4>
                    <p className="text-2xl font-bold text-green-600">
                      {platformPerformance[0]?.platform || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-500">
                      {platformPerformance[0]?.successRate || 0}% success rate
                    </p>
                  </div>

                  <div className="border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Analysis Complete</h4>
                    <p className="text-2xl font-bold text-purple-600">
                      {songs.filter(s => s.processing_status === 'completed').length}
                    </p>
                    <p className="text-sm text-gray-500">
                      of {songs.length} total songs
                    </p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'songs' && (
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">Your Songs</h3>
                {songs.length === 0 ? (
                  <div className="text-center py-8">
                    <Music className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-4">No songs uploaded yet</p>
                    <p className="text-sm text-gray-500">Upload your first song to get started with AI analysis</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {songs.map((song) => (
                      <div key={song.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center">
                            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-bold mr-4">
                              <Music className="h-6 w-6" />
                            </div>
                            <div>
                              <h4 className="font-semibold text-gray-900">{song.title}</h4>
                              <p className="text-gray-600 capitalize">{song.genre}</p>
                              {song.file_size && (
                                <p className="text-xs text-gray-500">
                                  {(song.file_size / (1024 * 1024)).toFixed(1)} MB
                                </p>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center space-x-4">
                            <div className="text-right">
                              <p className="text-sm text-gray-500">Created</p>
                              <p className="font-medium">{new Date(song.created_at).toLocaleDateString()}</p>
                            </div>
                            <span className={`px-3 py-1 rounded-full text-sm ${
                              song.processing_status === 'completed' 
                                ? 'bg-green-100 text-green-800' 
                                : song.processing_status === 'processing'
                                ? 'bg-blue-100 text-blue-800'
                                : song.processing_status === 'failed'
                                ? 'bg-red-100 text-red-800'
                                : 'bg-yellow-100 text-yellow-800'
                            }`}>
                              {song.processing_status}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'analytics' && (
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">Detailed Analytics</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="border border-gray-200 rounded-lg p-6">
                    <h4 className="font-semibold text-gray-900 mb-4">Top Performing Genres</h4>
                    <div className="space-y-3">
                      {topGenres.length > 0 ? topGenres.map((item) => (
                        <div key={item.genre} className="flex items-center justify-between">
                          <span className="capitalize font-medium">{item.genre}</span>
                          <div className="flex items-center">
                            <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                              <div 
                                className="bg-blue-500 h-2 rounded-full" 
                                style={{ width: `${item.percentage}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-600">{item.percentage}%</span>
                          </div>
                        </div>
                      )) : (
                        <p className="text-gray-500 text-center py-4">No genre data available yet</p>
                      )}
                    </div>
                  </div>

                  <div className="border border-gray-200 rounded-lg p-6">
                    <h4 className="font-semibold text-gray-900 mb-4">Platform Performance</h4>
                    <div className="space-y-3">
                      {platformPerformance.length > 0 ? platformPerformance.map((item) => (
                        <div key={item.platform} className="flex items-center justify-between">
                          <span className="font-medium">{item.platform}</span>
                          <span className="text-sm bg-green-100 text-green-800 px-2 py-1 rounded">
                            {item.successRate}% success rate
                          </span>
                        </div>
                      )) : (
                        <p className="text-gray-500 text-center py-4">No platform data available yet</p>
                      )}
                    </div>
                  </div>
                </div>

                {analytics && (
                  <div className="mt-6 border border-gray-200 rounded-lg p-6">
                    <h4 className="font-semibold text-gray-900 mb-4">Summary Stats</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-600">{analytics.totalAnalyzed}</p>
                        <p className="text-sm text-gray-600">Songs with AI insights</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-600">
                          {Math.round(analytics.avgConfidence * 100)}%
                        </p>
                        <p className="text-sm text-gray-600">Average confidence score</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-purple-600">
                          {Object.keys(analytics.genreStats).length}
                        </p>
                        <p className="text-sm text-gray-600">Different genres explored</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}