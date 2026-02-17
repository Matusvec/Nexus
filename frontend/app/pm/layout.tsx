import type { Metadata } from "next";
import { Fraunces, IBM_Plex_Sans } from "next/font/google";
import PMSidebar from "@/components/pm/layout/Sidebar";
import PipelineIndicator from "@/components/pm/PipelineIndicator";
import { PMQueryProvider } from "@/components/pm/PMQueryProvider";
import { KeyboardShortcutsProvider } from "@/components/pm/KeyboardShortcutsProvider";
import { Toaster } from "sonner";

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Nexus PM Pipeline",
  description: "Evidence-driven product management pipeline.",
};

export default function PMLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={`${plex.variable} ${fraunces.variable} pm-root`}>
      <div className="relative min-h-screen bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
        <div className="pm-backdrop" />
        <div className="pm-grid absolute inset-0 opacity-30" />
        <div className="relative flex min-h-screen">
          <PMSidebar />
          <main className="flex-1 px-4 py-6 sm:px-6 lg:px-8 pt-16 lg:pt-6">
            <div className="mb-6">
              <PipelineIndicator />
            </div>
            <PMQueryProvider>
              <KeyboardShortcutsProvider />
              {children}
            </PMQueryProvider>
          </main>
        </div>
      </div>
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: "pm-root",
          style: {
            fontFamily: "var(--font-sans)",
          },
        }}
      />
    </div>
  );
}
