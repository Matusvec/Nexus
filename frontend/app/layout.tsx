import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Nexus PM - Problem Management",
  description:
    "Evidence-driven problem discovery and management platform. Upload evidence, extract problems, cluster insights, and build your product roadmap.",
  keywords: [
    "product management",
    "problem discovery",
    "evidence-driven",
    "roadmap",
    "clustering",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <TooltipProvider delayDuration={0}>
          {children}
        </TooltipProvider>
      </body>
    </html>
  );
}
