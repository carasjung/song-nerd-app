// src/components/MarketingCopy.tsx
'use client'

import React, { useState } from 'react';
import { Copy, RefreshCw, Sparkles, Check } from 'lucide-react';

interface MarketingCopyProps {
  songData: {
    title: string;
    artist: string;
    genre: string;
    mood: string;
    tempo: number;
  };
}

export default function MarketingCopy({ songData }: MarketingCopyProps) {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Mock marketing copy suggestions
  const [marketingCopies] = useState([
    {
      platform: "Instagram",
      copy: `🎵 New drop alert! "${songData.title}" is giving me all the feels. Perfect ${songData.genre} vibes for your weekend playlist. Who's ready to vibe? #NewMusic #${songData.genre}Music #MoodBooster`,
      hashtags: ["#NewMusic", `#${songData.genre}Music`, "#MoodBooster", "#WeekendVibes"]
    },
    {
      platform: "TikTok",
      copy: `POV: You discover the perfect ${songData.genre} track.  "${songData.title}" hits different at ${songData.tempo} BPM. Use this sound for your next post!`,
      hashtags: ["#NewMusic", "#fyp", `#${songData.genre}`, "#VibeCheck"]
    },
    {
      platform: "Twitter",
      copy: `Just dropped "${songData.title}" and I'm not okay. This ${songData.genre} track is everything. Stream now and let me know your thoughts!`,
      hashtags: ["#NewRelease", `#${songData.genre}`, "#StreamNow"]
    },
    {
      platform: "Facebook",
      copy: `I'm so excited to share my latest single "${songData.title}" with you all! This ${songData.genre} track has been a labor of love, and I can't wait for you to experience the journey it takes you on. Perfect for those moments when you need that perfect soundtrack to your day. Give it a listen and share it with someone who needs to hear it!`,
      hashtags: [`#${songData.title}`, "#NewMusic", `#${songData.genre}`]
    }
  ]);

  const copyToClipboard = async (text: string, index: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const generateNewCopy = () => {
    setIsGenerating(true);
    // Simulate API call
    setTimeout(() => {
      setIsGenerating(false);
    }, 1500);
  };

  return (
    <div className="bg-white p-8 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Marketing Copy Suggestions</h2>
        <button
          onClick={generateNewCopy}
          disabled={isGenerating}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isGenerating ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4 mr-2" />
          )}
          {isGenerating ? 'Generating...' : 'Generate New'}
        </button>
      </div>

      <div className="space-y-6">
        {marketingCopies.map((item, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{item.platform}</h3>
              <button
                onClick={() => copyToClipboard(item.copy, index)}
                className="flex items-center px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                {copiedIndex === index ? (
                  <>
                    <Check className="h-4 w-4 mr-1 text-green-600" />
                    <span className="text-green-600 text-sm">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-1" />
                    <span className="text-sm">Copy</span>
                  </>
                )}
              </button>
            </div>
            
            <p className="text-gray-700 mb-4 leading-relaxed">{item.copy}</p>
            
            <div className="flex flex-wrap gap-2">
              {item.hashtags.map((hashtag, hashIndex) => (
                <span
                  key={hashIndex}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                >
                  {hashtag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
