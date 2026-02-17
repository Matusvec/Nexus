"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  AlertTriangle,
  Archive,
  FileText,
  LayoutDashboard,
  Layers,
  ListChecks,
  Map,
  Menu,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

const mainNav = [
  { href: "/pm", label: "Dashboard", icon: LayoutDashboard },
  { href: "/pm/evidence", label: "Evidence", icon: FileText },
  { href: "/pm/problems", label: "Problems", icon: AlertTriangle },
  { href: "/pm/clusters", label: "Clusters", icon: Layers },
  { href: "/pm/proposals", label: "Proposals", icon: Sparkles },
  { href: "/pm/tasks", label: "Tasks", icon: ListChecks },
  { href: "/pm/roadmap", label: "Roadmap", icon: Map },
];

const settingsNav = [
  { href: "/pm/settings", label: "Settings", icon: Settings },
  { href: "/pm/usage", label: "Usage", icon: Archive },
];

export default function PMSidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const renderLink = (item: (typeof mainNav)[0]) => {
    const isActive =
      item.href === "/pm"
        ? pathname === "/pm"
        : pathname.startsWith(item.href);
    return (
      <Link
        key={item.href}
        href={item.href}
        onClick={() => setMobileOpen(false)}
        className={cn(
          "flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors duration-150",
          isActive
            ? "bg-[hsl(var(--primary))] text-white shadow-sm"
            : "text-foreground/80 hover:bg-muted"
        )}
      >
        <item.icon className="h-4 w-4" strokeWidth={1.75} />
        <span className="font-medium">{item.label}</span>
      </Link>
    );
  };

  const sidebarContent = (
    <>
      {/* Brand */}
      <div className="flex h-16 items-center gap-3 px-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[hsl(var(--primary))] text-white">
          <span className="text-sm font-semibold">PM</span>
        </div>
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-muted-foreground">
            Nexus
          </p>
          <p className="text-base font-semibold leading-tight">Pipeline</p>
        </div>
      </div>

      {/* Main navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {mainNav.map(renderLink)}
        <div className="my-3 h-px bg-border" />
        {settingsNav.map(renderLink)}
      </nav>

      {/* Footer */}
      <div className="border-t border-border p-4 text-[11px] text-muted-foreground">
        Evidence to roadmap, end to end.
      </div>
    </>
  );

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="fixed left-4 top-4 z-50 flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card shadow-sm lg:hidden"
        aria-label="Open navigation"
      >
        <Menu className="h-5 w-5" strokeWidth={1.75} />
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile slide-in sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border bg-card transition-transform duration-200 ease-out lg:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <button
          onClick={() => setMobileOpen(false)}
          className="absolute right-3 top-5 flex h-8 w-8 items-center justify-center rounded-lg hover:bg-muted"
          aria-label="Close navigation"
        >
          <X className="h-4 w-4" />
        </button>
        {sidebarContent}
      </aside>

      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-64 flex-col border-r border-border bg-card/70 backdrop-blur lg:flex">
        {sidebarContent}
      </aside>
    </>
  );
}
