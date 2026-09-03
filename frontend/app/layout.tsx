import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ??
      'https://techgar-control.scli-xa2330670.chatgpt.site',
  ),
  title: 'TechGAR Control — Giám sát bãi xe',
  description: 'Bản đồ vận hành, điều hướng tài xế và kiểm soát Global ID cho hệ thống TechGAR.',
  openGraph: {
    title: 'TechGAR Control',
    description: 'Giám sát bãi xe và điều hướng tài xế theo Global ID.',
    type: 'website',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'TechGAR Control — Giám sát bãi xe theo Global ID' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'TechGAR Control',
    description: 'Giám sát bãi xe và điều hướng tài xế theo Global ID.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
