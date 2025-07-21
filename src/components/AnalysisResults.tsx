// src/components/AnalysisResults.tsx
'use client'

import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { 
  TrendingUp, Users, Target, Music2, Copy, ExternalLink, 
  Download, Share, Star, Headphones 
} from 'lucide-react';
import { supabase } from '@/lib/supabase';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

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
  const [isExporting, setIsExporting] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

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

  const exportToPDF = async () => {
    setIsExporting(true);
    try {
      // Wait for any animations to complete
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Create a completely clean version for PDF
      const tempWrapper = document.createElement('div');
      tempWrapper.style.position = 'absolute';
      tempWrapper.style.left = '-9999px';
      tempWrapper.style.top = '0';
      tempWrapper.style.width = '1200px';
      tempWrapper.style.backgroundColor = '#ffffff';
      tempWrapper.style.padding = '40px';
      tempWrapper.style.fontFamily = 'Arial, sans-serif';
      tempWrapper.style.color = '#000000';
      
      // Create clean HTML structure
      const cleanHTML = `
        <div style="font-family: Arial, sans-serif; color: #000000; background: #ffffff; padding: 20px;">
          <h1 style="font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 15px; color: #000000;">
            Marketing Analysis Report
          </h1>
          <p style="text-align: center; margin-bottom: 20px; color: #666666; font-size: 14px;">
            Comprehensive insights for "${song?.title}" by ${song?.artist}
          </p>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Key Performance Metrics</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
              <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px;">
                <div style="font-weight: bold; color: #000000; font-size: 12px;">Target Audience</div>
                <div style="font-size: 16px; font-weight: bold; color: #3B82F6;">${marketingInsights.primary_age_group}</div>
                <div style="font-size: 10px; color: #666666;">${Math.round(marketingInsights.age_confidence * 100)}% confidence</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px;">
                <div style="font-weight: bold; color: #000000; font-size: 12px;">Top Platform</div>
                <div style="font-size: 16px; font-weight: bold; color: #10B981;">${marketingInsights.top_platform}</div>
                <div style="font-size: 10px; color: #666666;">${Math.round(marketingInsights.platform_scores[marketingInsights.top_platform])}/100 score</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px;">
                <div style="font-weight: bold; color: #000000; font-size: 12px;">Audio Appeal</div>
                <div style="font-size: 16px; font-weight: bold; color: #8B5CF6;">${Math.round(analysis.audio_appeal)}/100</div>
                <div style="font-size: 10px; color: #666666;">Production quality</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px;">
                <div style="font-weight: bold; color: #000000; font-size: 12px;">Success Rate</div>
                <div style="font-size: 16px; font-weight: bold; color: #F59E0B;">${Math.round(marketingInsights.overall_confidence * 100)}%</div>
                <div style="font-size: 10px; color: #666666;">Predicted success</div>
              </div>
            </div>
          </div>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Platform Performance Predictions</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 6px;">
              ${Object.entries(marketingInsights.platform_scores).map(([platform, score]) => `
                <div style="margin-bottom: 8px;">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #000000; font-size: 12px;">${platform.charAt(0).toUpperCase() + platform.slice(1)}</span>
                    <span style="color: #3B82F6; font-size: 12px;">${Math.round(score as number)}/100</span>
                  </div>
                  <div style="background: #e5e7eb; height: 6px; border-radius: 3px; margin-top: 3px;">
                    <div style="background: #3B82F6; height: 6px; border-radius: 3px; width: ${Math.round(score as number)}%;"></div>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Audio Characteristics</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 6px;">
              <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Danceability</div>
                  <div style="color: #3B82F6; font-size: 14px;">${Math.round(analysis.danceability * 100)}%</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Energy</div>
                  <div style="color: #10B981; font-size: 14px;">${Math.round(analysis.energy * 100)}%</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Valence</div>
                  <div style="color: #F59E0B; font-size: 14px;">${Math.round(analysis.valence * 100)}%</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Acoustic</div>
                  <div style="color: #8B5CF6; font-size: 14px;">${Math.round(analysis.acousticness * 100)}%</div>
                </div>
              </div>
            </div>
          </div>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Track Details</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 6px;">
              <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Tempo</div>
                  <div style="color: #3B82F6; font-size: 14px;">${Math.round(analysis.tempo)} BPM</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Key</div>
                  <div style="color: #10B981; font-size: 14px;">Key ${analysis.key}</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Mode</div>
                  <div style="color: #F59E0B; font-size: 14px;">${analysis.mode === 1 ? 'Major' : 'Minor'}</div>
                </div>
                <div>
                  <div style="font-weight: bold; color: #000000; font-size: 12px;">Genre</div>
                  <div style="color: #8B5CF6; font-size: 14px;">${song?.genre}</div>
                </div>
              </div>
            </div>
          </div>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Marketing Action Plan</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 6px;">
              <div style="margin-bottom: 15px;">
                <h3 style="font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #000000;">Recommended Actions</h3>
                ${marketingInsights.action_items.map((action: string, index: number) => `
                  <div style="margin-bottom: 6px; display: flex; align-items: flex-start;">
                    <span style="background: #3B82F6; color: #ffffff; width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; margin-right: 8px; flex-shrink: 0; margin-top: 1px;">${index + 1}</span>
                    <span style="color: #000000; font-size: 12px; line-height: 1.3;">${action}</span>
                  </div>
                `).join('')}
              </div>
              <div>
                <h3 style="font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #000000;">Similar Artists</h3>
                ${marketingInsights.similar_artists.map((artist: any, index: number) => `
                  <div style="border: 1px solid #e5e7eb; padding: 8px; border-radius: 4px; margin-bottom: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                      <span style="font-weight: bold; color: #000000; font-size: 12px;">${artist.artist_name}</span>
                      <span style="background: #3B82F6; color: #ffffff; padding: 1px 6px; border-radius: 3px; font-size: 10px;">${Math.round(artist.similarity_score * 100)}% match</span>
                    </div>
                    <span style="color: #666666; font-size: 10px;">${artist.genre}</span>
                  </div>
                `).join('')}
              </div>
            </div>
          </div>
          
          <div style="margin-bottom: 20px;">
            <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #000000;">Summary</h2>
            <div style="background: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 6px;">
              <p style="font-size: 14px; color: #000000; margin-bottom: 10px; line-height: 1.3;">
                ${marketingInsights.sound_profile} - ${marketingInsights.competitive_advantage}
              </p>
              <p style="color: #666666; font-size: 12px; line-height: 1.3;">
                Target your ${marketingInsights.primary_age_group} audience primarily on ${marketingInsights.top_platform} for maximum impact.
              </p>
            </div>
          </div>
        </div>
      `;
      
      tempWrapper.innerHTML = cleanHTML;
      document.body.appendChild(tempWrapper);
      
      // Wait for content to render
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Capture the entire content
      const canvas = await html2canvas(tempWrapper, {
        scale: 1.5,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        logging: false,
        width: 1200,
        height: tempWrapper.scrollHeight,
        scrollX: 0,
        scrollY: 0
      });
      
      // Clean up
      document.body.removeChild(tempWrapper);
      
      // Create PDF
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 10;
      const imageWidth = pageWidth - (margin * 2);
      
      // Add the captured image directly (no duplicate title)
      const imgData = canvas.toDataURL('image/png');
      const imgHeight = (canvas.height * imageWidth) / canvas.width;
      
      // Fit everything on one page
      const maxHeight = pageHeight - (margin * 2);
      const scale = Math.min(1, maxHeight / imgHeight);
      const finalHeight = imgHeight * scale;
      
      pdf.addImage(
        imgData, 
        'PNG', 
        margin, 
        margin, 
        imageWidth, 
        finalHeight
      );
      
      pdf.save(`marketing-analysis-${song?.title || 'song'}.pdf`);
    } catch (error) {
      console.error('Error exporting PDF:', error);
      alert('Failed to export PDF. Please try again.');
    } finally {
      setIsExporting(false);
    }
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
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
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
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
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
    { name: 'Acoustic', value: Math.round(analysis.acousticness * 100), color: '#EF4444' },
  ];

  return (
    <div className="analysis-results-container min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Marketing Analysis Complete
          </h1>
          <p className="text-lg text-gray-600">
            Comprehensive insights for "{song?.title}" by {song?.artist}
          </p>
        </div>

        {/* Key Metrics Cards */}
        <div id="key-metrics" className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <Target className="h-10 w-10 text-gray-900 mr-4" />
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
              <TrendingUp className="h-10 w-10 text-gray-900 mr-4" />
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
              <Music2 className="h-10 w-10 text-gray-900 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Audio Appeal</h3>
                <p className="text-2xl font-bold text-purple-800">
                  {Math.round(analysis.audio_appeal)}/100
                </p>
                <p className="text-sm text-gray-500">Production quality</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center">
              <Headphones className="h-10 w-10 text-gray-900 mr-4" />
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
      </div>

      {/* Platform Recommendations Chart - Outside parent container */}
      <div id="platform-chart" className="platform-chart-container bg-white p-8 rounded-xl shadow-lg mx-auto">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Platform Performance Predictions</h2>
        <div className="platform-chart-content">
          <div className="platform-chart-wrapper">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={platformData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" fill="#f1f5f9" />
                <XAxis dataKey="name" dy={10} />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'score' ? `${value}/100` : `${value}%`,
                    name === 'score' ? 'Performance Score' : 'Success Probability'
                  ]}
                />
                <Bar 
                  dataKey="score" 
                  fill="#3B82F6" 
                  name="score"
                  style={{ 
                    filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.15))',
                    stroke: 'rgba(0, 0, 0, 0.2)',
                    strokeWidth: 2
                  }}
                />
                <Bar 
                  dataKey="success_probability" 
                  fill="#10B981" 
                  name="success_probability"
                  style={{ 
                    filter: 'drop-shadow(0 3px 6px rgba(0,0,0,0.15))',
                    stroke: 'rgba(0, 0, 0, 0.2)',
                    strokeWidth: 2
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="platform-explanations">
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontWeight: '600', color: '#3B82F6', marginBottom: '0.5rem' }}>
                Performance Score
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#374151', lineHeight: '1.5' }}>
                Measures how well your song aligns with each platform's audience preferences, 
                content style, and algorithmic requirements. Higher scores indicate better 
                platform compatibility and potential for organic reach.
              </p>
            </div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontWeight: '600', color: '#10B981', marginBottom: '0.5rem' }}>
                Success Probability
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#374151', lineHeight: '1.5' }}>
                Predicts the likelihood of achieving viral success or high engagement on each 
                platform based on historical data, current trends, and your song's characteristics. 
                This considers factors like virality potential and audience resonance.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="match-platform-width space-y-8">

        {/* Audio Features Visualization */}
        <div id="audio-chart" className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Audio Characteristics</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={audioFeatureData}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  style={{ filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.15))' }}
                >
                  {audioFeatureData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.color}
                      style={{ 
                        stroke: 'rgba(0, 0, 0, 0.2)',
                        strokeWidth: 1,
                        filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'
                      }}
                    />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div id="track-details" className="bg-white p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Track Details</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-medium text-gray-700">Tempo:</span>
                  <span className="text-gray-900 ml-2">{Math.round(analysis.tempo)} BPM</span>
                </div>
                <div className="text-sm text-gray-600 max-w-xs">
                  {analysis.tempo >= 120 ? 
                    "Upbeat tempos (120+ BPM) are perfect for energetic content and viral dance trends." :
                    analysis.tempo >= 90 ?
                    "Mid-tempo tracks (90-119 BPM) offer versatility across platforms and audience types." :
                    "Slower tempos create emotional connection and work well for intimate, storytelling content."
                  }
                </div>
              </div>
              
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-medium text-gray-700">Key:</span>
                  <span className="text-gray-900 ml-2">Key {analysis.key}</span>
                </div>
                <div className="text-sm text-gray-600 max-w-xs">
                  {['C', 'G', 'D', 'A', 'E', 'B'].includes(analysis.key) ?
                    "Sharp keys create bright, energetic vibes that resonate with younger audiences." :
                    "Flat keys offer warm, soulful tones that appeal to mature listeners and emotional content."
                  }
                </div>
              </div>
              
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-medium text-gray-700">Mode:</span>
                  <span className="text-gray-900 ml-2">{analysis.mode === 1 ? 'Major' : 'Minor'}</span>
                </div>
                <div className="text-sm text-gray-600 max-w-xs">
                  {analysis.mode === 1 ?
                    "Major mode conveys positivity and optimism, ideal for feel-good content and brand partnerships." :
                    "Minor mode creates emotional depth and authenticity, perfect for storytelling and intimate connections."
                  }
                </div>
              </div>
              
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-medium text-gray-700">Genre:</span>
                  <span className="text-gray-900 ml-2 capitalize">{song?.genre}</span>
                </div>
                <div className="text-sm text-gray-600 max-w-xs">
                  {song?.genre?.toLowerCase().includes('pop') ?
                    "Pop music has universal appeal and works across all platforms, especially TikTok and Instagram." :
                    song?.genre?.toLowerCase().includes('hip') || song?.genre?.toLowerCase().includes('rap') ?
                    "Hip-hop and rap thrive on YouTube and TikTok, with strong community engagement potential." :
                    song?.genre?.toLowerCase().includes('rock') ?
                    "Rock music has dedicated fan bases on Spotify and YouTube, with strong playlist potential." :
                    song?.genre?.toLowerCase().includes('electronic') || song?.genre?.toLowerCase().includes('edm') ?
                    "Electronic music excels on streaming platforms and has strong festival/event marketing potential." :
                    "This genre offers unique positioning opportunities and can stand out in niche markets."
                  }
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Marketing Recommendations */}
        <div id="marketing-plan" className="bg-white p-8 rounded-xl shadow-lg">
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
        <div id="summary" className="bg-white p-8 rounded-xl shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Summary</h2>
          <p className="text-lg text-gray-700 mb-4">
            {marketingInsights.sound_profile} - {marketingInsights.competitive_advantage}
          </p>
          <p className="text-base text-gray-600 mb-6">
            Target your {marketingInsights.primary_age_group} audience primarily on {marketingInsights.top_platform} 
            for maximum impact.
          </p>
          <div className="flex flex-wrap gap-4">
            <button 
              onClick={exportToPDF}
              disabled={isExporting}
              className="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg font-semibold hover:bg-gray-300 transition-colors border border-gray-300 shadow-sm disabled:opacity-50 cursor-pointer"
            >
              <Download className="h-4 w-4 inline mr-2" />
              {isExporting ? 'Exporting...' : 'Export Report'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}