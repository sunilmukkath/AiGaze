"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BrandLogo } from "./BrandLogo";
import { AI_GAZE_STUDIO_URL } from "@/lib/product";

interface ETHeaderProps {
  variant?: "public" | "app";
}

const NAV_LINKS = [
  { href: "/#features", label: "Features" },
  { href: "/methodology", label: "Method" },
  { href: "/pricing", label: "Pricing" },
  { href: "/#studio", label: "Studio" },
];

export function ETHeader({ variant = "public" }: ETHeaderProps) {
  const isPublic = variant === "public";
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (!isPublic) return;
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [isPublic]);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-300 ${
        isPublic ? `site-header ${scrolled ? "site-header--solid" : ""}` : "border-b border-white/10"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2.5 shrink-0 group" aria-label="AI Gaze home">
          <BrandLogo height={48} priority />
        </Link>

        {isPublic && (
          <nav className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map((link) => (
              <Link key={link.href} href={link.href} className="nav-link px-3.5 py-2 text-sm font-medium rounded-lg">
                {link.label}
              </Link>
            ))}
          </nav>
        )}

        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <a
            href={AI_GAZE_STUDIO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link text-sm px-2 sm:px-3 py-2 hidden sm:inline"
          >
            Sign In
          </a>
          <a
            href={AI_GAZE_STUDIO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="et-btn-primary text-sm !py-2.5 !px-5"
          >
            Launch Studio
          </a>
        </div>
      </div>
    </header>
  );
}
