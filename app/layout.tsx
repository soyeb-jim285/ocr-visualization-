import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SmoothScrollProvider } from "@/components/providers/SmoothScrollProvider";
import { ModelProvider } from "@/components/providers/ModelProvider";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Neural Network X-Ray | Interactive CNN Visualization",
  description:
    "An interactive visualization of how convolutional neural networks recognize handwritten characters. Draw a letter or digit and watch every layer of the network process it in real-time.",
  keywords: [
    "neural network",
    "CNN",
    "visualization",
    "machine learning",
    "EMNIST",
    "deep learning",
    "OCR",
  ],
  openGraph: {
    title: "Neural Network X-Ray",
    description:
      "Draw a character and watch every layer of a CNN process it in real time — from raw pixels to confident prediction.",
    type: "website",
    siteName: "Neural Network X-Ray",
  },
  twitter: {
    card: "summary_large_image",
    title: "Neural Network X-Ray",
    description:
      "Interactive visualization of how CNNs recognize handwritten characters. Draw anything and see 13 layers process it live.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <TooltipProvider delayDuration={0}>
          <SmoothScrollProvider>
            <ModelProvider>{children}</ModelProvider>
          </SmoothScrollProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
