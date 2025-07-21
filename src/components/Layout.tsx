// src/components/Layout.tsx
'use client'

import React, { useState, useEffect } from 'react';
import { User, BarChart3, Upload, Menu, X } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';

interface LayoutProps {
  children: React.ReactNode;
  currentPage?: string;
}

export default function Layout({ children, currentPage = 'upload' }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  const navigation = [
    { name: 'Upload', href: '/', icon: Upload, current: currentPage === 'upload' },
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3, current: currentPage === 'dashboard' },
    { name: 'Profile', href: '/profile', icon: User, current: currentPage === 'profile' },
  ];

  // Handle scroll events for top bar animation
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 50);
    };

    document.addEventListener('scroll', handleScroll, { passive: true });
    return () => document.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="sidebar-content">
          {/* Sidebar header with bars button */}
          <div className="sidebar-header">
            <button
              type="button"
              className="p-2 text-white hover:text-gray-300 transition-colors cursor-pointer"
              onClick={() => setSidebarOpen(false)}
            >
              <i className="fa-solid fa-xmark text-xl" style={{ color: '#ffffff' }}></i>
            </button>
          </div>
          
          <div className="flex flex-1 flex-col overflow-y-auto">
            <nav className="mt-4 flex-1 space-y-1 px-2">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                    item.current
                      ? 'bg-gray-800 text-white border border-gray-600'
                      : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                  }`}
                  onClick={() => setSidebarOpen(false)}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className={`main-content ${sidebarOpen ? 'main-content-shifted' : ''}`}>
        {/* Top bar */}
        <div className={`top-bar transition-all duration-300 ease-in-out ${
          isScrolled ? '!h-11' : '!h-24'
        }`}>
          <button
            type="button"
            className={`p-2 text-white hover:text-gray-300 transition-colors cursor-pointer ${sidebarOpen ? 'hidden' : 'block'}`}
            onClick={() => setSidebarOpen(true)}
          >
            <i className="fa-solid fa-bars text-xl"></i>
          </button>
          
          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="flex items-center justify-center flex-1">
              <Image 
                src="/logo.png" 
                alt="Logo" 
                width={180} 
                height={60} 
                className={`w-auto transition-all duration-300 ease-in-out ${
                  isScrolled ? 'h-10' : 'h-14'
                }`} 
              />
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="bg-white min-h-screen">
          {children}
        </main>
      </div>
    </div>
  );
}
