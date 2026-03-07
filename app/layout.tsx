import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Will It Fit? | FitMyGPU",
  description: "Estimate GPU VRAM for model inference and training.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
