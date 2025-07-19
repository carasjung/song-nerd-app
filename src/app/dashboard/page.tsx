// src/app/dashboard/page.tsx
'use client'

import Layout from '@/components/Layout';
import ArtistProfile from '@/components/ArtistProfile';

export default function Dashboard() {
  return (
    <Layout currentPage="dashboard">
      <ArtistProfile />
    </Layout>
  );
}
