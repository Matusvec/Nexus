"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  Brain,
  LayoutGrid,
  FileText,
  MessageSquare,
  Settings,
  ChevronLeft,
  ChevronRight,
  Search,
  Plus,
  Command,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { useUIStore, useChatStore, useDocumentsStore } from "@/lib/store";
import { cn } from "@/lib/utils";

const navItems = [
  {
    href: "/workspace",
    icon: LayoutGrid,
    label: "Canvas",
    shortcut: "⌘1",
  },
  {
    href: "/documents",
    icon: FileText,
    label: "Documents",
    shortcut: "⌘2",
  },
];

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

export default function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const pathname = usePathname();
  const { setUploadModalOpen, setSearchOpen, setSettingsModalOpen } = useUIStore();
  const { isSidebarOpen, toggleSidebar } = useChatStore();
  const { documents } = useDocumentsStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 240 }}
      transition={{ duration: 0.2 }}
      className="h-full bg-card border-r border-border flex flex-col"
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-border">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-nexus-gradient flex items-center justify-center flex-shrink-0">
            <Brain className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-lg font-bold"
            >
              Nexus
            </motion.span>
          )}
        </Link>
        {!collapsed && (
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onToggle}>
            <ChevronLeft className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Search Button */}
      <div className="p-3">
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-muted-foreground hover:text-foreground",
            collapsed && "justify-center px-0"
          )}
          onClick={() => setSearchOpen(true)}
        >
          <Search className="w-4 h-4" />
          {!collapsed && (
            <>
              <span className="ml-2 flex-1 text-left">Search...</span>
              <kbd className="ml-auto pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
                <span className="text-xs">⌘</span>K
              </kbd>
            </>
          )}
        </Button>
      </div>

      {/* Quick Actions */}
      <div className="px-3 mb-2">
        <Button
          variant="default"
          className={cn("w-full", collapsed && "px-0")}
          onClick={() => setUploadModalOpen(true)}
        >
          <Plus className="w-4 h-4" />
          {!collapsed && <span className="ml-2">Upload Document</span>}
        </Button>
      </div>

      <Separator className="mx-3" />

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {!collapsed && (
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2 px-2">
            Navigation
          </p>
        )}
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Tooltip key={item.href}>
              <TooltipTrigger asChild>
                <Link href={item.href}>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "w-full justify-start",
                      collapsed && "justify-center px-0",
                      isActive && "bg-primary/10 text-primary"
                    )}
                  >
                    <item.icon className="w-4 h-4" />
                    {!collapsed && (
                      <>
                        <span className="ml-2 flex-1 text-left">{item.label}</span>
                        <kbd className="ml-auto text-[10px] text-muted-foreground">
                          {item.shortcut}
                        </kbd>
                      </>
                    )}
                  </Button>
                </Link>
              </TooltipTrigger>
              {collapsed && <TooltipContent side="right">{item.label}</TooltipContent>}
            </Tooltip>
          );
        })}

        {/* Chat Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isSidebarOpen ? "secondary" : "ghost"}
              className={cn(
                "w-full justify-start",
                collapsed && "justify-center px-0",
                isSidebarOpen && "bg-primary/10 text-primary"
              )}
              onClick={toggleSidebar}
            >
              <MessageSquare className="w-4 h-4" />
              {!collapsed && (
                <>
                  <span className="ml-2 flex-1 text-left">AI Chat</span>
                  <Badge variant="info" className="text-xs">
                    Active
                  </Badge>
                </>
              )}
            </Button>
          </TooltipTrigger>
          {collapsed && <TooltipContent side="right">AI Chat</TooltipContent>}
        </Tooltip>
      </nav>

      {/* Stats */}
      {!collapsed && (
        <div className="p-3 mx-3 mb-3 rounded-lg bg-muted/50">
          <p className="text-xs text-muted-foreground mb-2">Knowledge Base</p>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div>
              <p className="text-lg font-bold text-primary">{documents.length}</p>
              <p className="text-xs text-muted-foreground">Documents</p>
            </div>
            <div>
              <p className="text-lg font-bold text-nexus-purple">3</p>
              <p className="text-xs text-muted-foreground">Layers</p>
            </div>
          </div>
        </div>
      )}

      {/* Settings */}
      <div className="p-3 border-t border-border">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start",
                collapsed && "justify-center px-0"
              )}
              onClick={() => setSettingsModalOpen(true)}
            >
              <Settings className="w-4 h-4" />
              {!collapsed && <span className="ml-2">Settings</span>}
            </Button>
          </TooltipTrigger>
          {collapsed && <TooltipContent side="right">Settings</TooltipContent>}
        </Tooltip>
      </div>

      {/* Collapse Toggle (when collapsed) */}
      {collapsed && (
        <div className="p-3 border-t border-border">
          <Button
            variant="ghost"
            size="icon"
            className="w-full"
            onClick={onToggle}
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      )}
    </motion.aside>
  );
}
