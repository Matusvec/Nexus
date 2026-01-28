"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  FileText,
  Folder,
  Clock,
  ArrowRight,
  Command,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useUIStore, useDocumentsStore, useCanvasStore } from "@/lib/store";
import { cn } from "@/lib/utils";

interface SearchResult {
  id: string;
  type: "document" | "group" | "recent";
  title: string;
  subtitle?: string;
  icon: typeof FileText;
}

export default function SearchCommand() {
  const { isSearchOpen, setSearchOpen } = useUIStore();
  const { documents } = useDocumentsStore();
  const { groups } = useCanvasStore();
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Generate search results
  const results: SearchResult[] = [];

  // Add documents
  documents
    .filter(
      (doc) =>
        !query || doc.filename.toLowerCase().includes(query.toLowerCase())
    )
    .slice(0, 5)
    .forEach((doc) => {
      results.push({
        id: doc.id,
        type: "document",
        title: doc.filename,
        subtitle: `${doc.chunkCount} chunks`,
        icon: FileText,
      });
    });

  // Add groups
  groups
    .filter(
      (group) =>
        !query || group.name.toLowerCase().includes(query.toLowerCase())
    )
    .slice(0, 3)
    .forEach((group) => {
      results.push({
        id: group.id,
        type: "group",
        title: group.name,
        subtitle: `${group.documentIds.length} documents`,
        icon: Folder,
      });
    });

  // Add recent searches (mock)
  if (!query) {
    const recentSearches = [
      "RAPTOR tree structure",
      "Motor specifications",
      "Python implementation",
    ];
    recentSearches.forEach((search, idx) => {
      results.push({
        id: `recent-${idx}`,
        type: "recent",
        title: search,
        icon: Clock,
      });
    });
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isSearchOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (results[selectedIndex]) {
            handleSelect(results[selectedIndex]);
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isSearchOpen, results, selectedIndex]);

  // Reset on open
  useEffect(() => {
    if (isSearchOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isSearchOpen]);

  // Global keyboard shortcut
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [setSearchOpen]);

  const handleSelect = (result: SearchResult) => {
    console.log("Selected:", result);
    setSearchOpen(false);
    // Handle navigation based on result type
  };

  return (
    <Dialog open={isSearchOpen} onOpenChange={setSearchOpen}>
      <DialogContent className="sm:max-w-xl p-0 overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center border-b border-border px-4">
          <Search className="w-4 h-4 text-muted-foreground" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search documents, groups, or ask a question..."
            className="border-0 focus-visible:ring-0 text-base py-6"
          />
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <ScrollArea className="max-h-[400px]">
          <div className="p-2">
            {results.length === 0 ? (
              <div className="py-8 text-center">
                <p className="text-muted-foreground">No results found</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Try a different search term
                </p>
              </div>
            ) : (
              <>
                {/* Documents Section */}
                {results.filter((r) => r.type === "document").length > 0 && (
                  <div className="mb-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider px-2 py-1">
                      Documents
                    </p>
                    {results
                      .filter((r) => r.type === "document")
                      .map((result, idx) => {
                        const globalIdx = results.indexOf(result);
                        return (
                          <SearchResultItem
                            key={result.id}
                            result={result}
                            isSelected={selectedIndex === globalIdx}
                            onSelect={() => handleSelect(result)}
                            onHover={() => setSelectedIndex(globalIdx)}
                          />
                        );
                      })}
                  </div>
                )}

                {/* Groups Section */}
                {results.filter((r) => r.type === "group").length > 0 && (
                  <div className="mb-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider px-2 py-1">
                      Groups
                    </p>
                    {results
                      .filter((r) => r.type === "group")
                      .map((result) => {
                        const globalIdx = results.indexOf(result);
                        return (
                          <SearchResultItem
                            key={result.id}
                            result={result}
                            isSelected={selectedIndex === globalIdx}
                            onSelect={() => handleSelect(result)}
                            onHover={() => setSelectedIndex(globalIdx)}
                          />
                        );
                      })}
                  </div>
                )}

                {/* Recent Section */}
                {results.filter((r) => r.type === "recent").length > 0 && (
                  <div className="mb-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider px-2 py-1">
                      Recent Searches
                    </p>
                    {results
                      .filter((r) => r.type === "recent")
                      .map((result) => {
                        const globalIdx = results.indexOf(result);
                        return (
                          <SearchResultItem
                            key={result.id}
                            result={result}
                            isSelected={selectedIndex === globalIdx}
                            onSelect={() => handleSelect(result)}
                            onHover={() => setSelectedIndex(globalIdx)}
                          />
                        );
                      })}
                  </div>
                )}
              </>
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/50 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-muted border">↑↓</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-muted border">↵</kbd>
              Select
            </span>
          </div>
          <span className="flex items-center gap-1">
            <Command className="w-3 h-3" />K to open
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function SearchResultItem({
  result,
  isSelected,
  onSelect,
  onHover,
}: {
  result: SearchResult;
  isSelected: boolean;
  onSelect: () => void;
  onHover: () => void;
}) {
  return (
    <motion.button
      onClick={onSelect}
      onMouseEnter={onHover}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors",
        isSelected ? "bg-primary/10 text-primary" : "hover:bg-muted"
      )}
      whileTap={{ scale: 0.98 }}
    >
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center",
          isSelected ? "bg-primary/20" : "bg-muted"
        )}
      >
        <result.icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm truncate">{result.title}</p>
        {result.subtitle && (
          <p className="text-xs text-muted-foreground">{result.subtitle}</p>
        )}
      </div>
      {isSelected && <ArrowRight className="w-4 h-4 flex-shrink-0" />}
    </motion.button>
  );
}
