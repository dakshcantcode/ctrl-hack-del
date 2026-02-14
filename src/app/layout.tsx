/**
 * NeuroSketch — Root Layout
 * Application shell with dark theme and custom fonts.
 *
 * @module app/layout
 */

import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "NeuroSketch — Parkinson's Early Detection",
  description:
    "A high-fidelity screening tool for early Parkinson's detection using Archimedes spiral analysis and 3D neural visualization.",
  keywords: [
    "Parkinson's detection",
    "spiral analysis",
    "tremor screening",
    "neurology",
    "motor function",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black text-zinc-100`}
      >
        {children}
      </body>
    </html>
  );
}
