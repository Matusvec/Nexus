// ============================================
// PageTransition — Framer Motion page entrance — Strategy §6
// ============================================
// Wraps page content with fade + slide-up animation.
// Respects prefers-reduced-motion.

"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface PageTransitionProps {
  children: ReactNode;
  className?: string;
}

const variants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -4 },
};

export function PageTransition({ children, className }: PageTransitionProps) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={variants}
      transition={{
        duration: 0.2,
        ease: [0.16, 1, 0.3, 1], // Vercel-style spring
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
