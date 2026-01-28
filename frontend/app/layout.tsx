import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Nexus - Your Personal Team of AI Research Specialists",
  description:
    "Transform document chaos into an intelligent, collaborative AI workspace. Organize 1000s of documents, query with human-like understanding, and delegate work to specialist AI personas.",
  keywords: [
    "AI",
    "RAG",
    "RAPTOR",
    "document management",
    "research assistant",
    "knowledge base",
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
