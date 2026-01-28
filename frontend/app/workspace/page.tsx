"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import Sidebar from "@/components/layout/Sidebar";
import ChatInterface from "@/components/chat/ChatInterface";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { useChatStore } from "@/lib/store";
import { cn } from "@/lib/utils";

// Dynamically import canvas to avoid SSR issues with React Flow
const WorkspaceCanvas = dynamic(
  () => import("@/components/canvas/WorkspaceCanvas"),
  { ssr: false }
);

export default function WorkspacePage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { isSidebarOpen } = useChatStore();

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      {/* Left Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Canvas Area */}
        <div className="flex-1 relative">
          <WorkspaceCanvas />
        </div>

        {/* Chat Sidebar */}
        <AnimatePresence>
          {isSidebarOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 400, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="h-full border-l border-border overflow-hidden"
            >
              <ChatInterface />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Modals */}
      <UploadModal />
      <SearchCommand />
    </div>
  );
}
