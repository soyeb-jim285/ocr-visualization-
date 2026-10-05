import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Newsreader, Noto_Sans_Bengali } from "next/font/google";
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

const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin"],
  axes: ["opsz"],
  style: ["normal", "italic"],
  display: "swap",
});

// Fallback for the Bengali glyphs in the hero cycler
const notoBengali = Noto_Sans_Bengali({
  variable: "--font-bn",
  subsets: ["bengali"],
  display: "swap",
});

export const viewport: Viewport = { themeColor: "#06080b", colorScheme: "dark", viewportFit: "cover" };

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
        className={`${geistSans.variable} ${geistMono.variable} ${newsreader.variable} ${notoBengali.variable} antialiased`}
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
