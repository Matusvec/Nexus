"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Archive,
  BookOpen,
  ClipboardList,
  FileText,
  Layers,
  ListChecks,
  Sliders,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/pm", label: "Dashboard", icon: Activity },
  { href: "/pm/evidence", label: "Evidence", icon: FileText },
  { href: "/pm/problems", label: "Problems", icon: BookOpen },
  { href: "/pm/clusters", label: "Clusters", icon: Layers },
  { href: "/pm/proposals", label: "Proposals", icon: Sparkles },
  { href: "/pm/tasks", label: "Tasks", icon: ClipboardList },
  { href: "/pm/roadmap", label: "Roadmap", icon: ListChecks },
  { href: "/pm/settings", label: "Settings", icon: Sliders },
  { href: "/pm/usage", label: "Usage", icon: Archive },
];

export default function PMSidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 flex-col border-r border-border bg-card/70 backdrop-blur">
      <div className="flex h-16 items-center gap-3 px-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[hsl(var(--primary))] text-white">
          <span className="text-sm font-semibold">PM</span>
        </div>
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
            Nexus
          </p>
          <p className="text-lg font-semibold">Pipeline</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition",
                isActive
                  ? "bg-[hsl(var(--primary))] text-white shadow-sm"
                  : "text-foreground/80 hover:bg-muted"
              )}
            >
              <item.icon className="h-4 w-4" />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border p-4 text-xs text-muted-foreground">
        Evidence to roadmap, end to end.
      </div>
    </aside>
  );
}
