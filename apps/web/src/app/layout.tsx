import type { Metadata } from "next";
import { DM_Sans, DM_Mono, Outfit } from "next/font/google";
import "./globals.css";

const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-dm-sans" });
const dmMono = DM_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-dm-mono" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "AI Gaze™ — Predictive Eye Tracking | Elastic Tree",
  description:
    "AI Gaze™ predicts visual attention on packaging, shelves, and ads with 92% accuracy — heat maps, gaze path, and branded reports without hardware eye-tracking.",
  themeColor: "#0a1f4a",
  icons: {
    icon: [
      { url: "/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/favicon.png", sizes: "256x256", type: "image/png" },
      { url: "/aigaze-icon.png", sizes: "256x256", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${dmSans.variable} ${dmMono.variable} ${outfit.variable} h-full`}>
      <body className="min-h-full antialiased page-spectrum">{children}</body>
    </html>
  );
}
