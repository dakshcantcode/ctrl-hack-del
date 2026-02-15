/**
 * ScrollReveal — Scroll-triggered entrance wrapper
 *
 * Uses Framer Motion + IntersectionObserver (via whileInView)
 * to fade-in and scale-up children when they enter the viewport,
 * and fade-out when they leave.
 *
 * Supports directional variants: "up" (default) or "left".
 *
 * @module components/ui/scroll-reveal
 */

"use client";

import React from "react";
import { motion, type Variants } from "framer-motion";

interface ScrollRevealProps {
  children: React.ReactNode;
  className?: string;
  /** Delay in seconds before the entrance animation starts */
  delay?: number;
  /** Viewport margin for early trigger (default: "-60px") */
  margin?: string;
  /** Direction the element enters from (default: "up") */
  direction?: "up" | "left";
}

const upVariants: Variants = {
  hidden: { opacity: 0, y: 32, scale: 0.95 },
  visible: { opacity: 1, y: 0, scale: 1 },
};

const leftVariants: Variants = {
  hidden: { opacity: 0, x: -48 },
  visible: { opacity: 1, x: 0 },
};

export default function ScrollReveal({
  children,
  className,
  delay = 0,
  margin = "-60px",
  direction = "up",
}: ScrollRevealProps) {
  return (
    <motion.div
      variants={direction === "left" ? leftVariants : upVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: false, margin }}
      transition={{
        type: "spring",
        stiffness: 80,
        damping: 20,
        delay,
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
