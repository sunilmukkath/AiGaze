import Link from "next/link";
import Image from "next/image";
import { BrandLogo } from "./BrandLogo";

export function ETFooter() {
  return (
    <footer className="border-t border-white/10 bg-[#0a1f4a]/55 backdrop-blur-xl text-[var(--text-muted)] py-16">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-10 mb-12">
          <div className="md:col-span-1">
            <Link href="/" className="inline-flex items-center gap-2.5 mb-4">
              <BrandLogo height={36} />
            </Link>
            <p className="text-sm font-semibold text-[var(--amber-light)] mb-1">
              Predictive Eye Tracking
            </p>
            <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
              See what gets attention in the first 3 seconds — without hardware eye-tracking.
            </p>
            <p className="text-sm mt-4">
              <a href="mailto:sunil@elastictree.com" className="hover:text-[var(--amber)] transition">
                sunil@elastictree.com
              </a>
            </p>
            <p className="text-sm mt-1">Sales & Support: +91 98408 50057</p>
          </div>
          <div>
            <p className="text-[var(--text-primary)] text-sm font-semibold mb-3">Product</p>
            <ul className="space-y-2 text-sm text-[var(--text-secondary)]">
              <li><Link href="/#features" className="hover:text-[var(--amber)] transition">Features</Link></li>
              <li><Link href="/methodology" className="hover:text-[var(--amber)] transition">Method</Link></li>
              <li><Link href="/pricing" className="hover:text-[var(--amber)] transition">Pricing</Link></li>
              <li><Link href="/#studio" className="hover:text-[var(--amber)] transition">Studio</Link></li>
            </ul>
          </div>
          <div>
            <p className="text-[var(--text-primary)] text-sm font-semibold mb-3">Chennai</p>
            <p className="text-sm leading-relaxed">
              3B, Krshnika Apartments, 1/26, Avenue Rd,
              <br />
              Nungambakkam, Chennai 600034
            </p>
          </div>
          <div>
            <p className="text-[var(--text-primary)] text-sm font-semibold mb-3">An Elastic Tree product</p>
            <a
              href="https://www.elastictree.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 hover:opacity-90 transition"
            >
              <Image
                src="/elastic-tree-logo.png"
                alt="Elastic Tree"
                width={140}
                height={32}
                className="object-contain h-7 w-auto"
              />
            </a>
            <p className="text-sm mt-3 leading-relaxed text-[var(--text-secondary)]">
              Smart Decisions, Simply Made.
            </p>
            <p className="text-sm mt-2">
              <a
                href="https://www.elastictree.com/"
                className="hover:text-[var(--amber)] transition"
                target="_blank"
                rel="noopener noreferrer"
              >
                elastictree.com
              </a>
            </p>
          </div>
        </div>
        <div className="border-t border-white/10 pt-6 flex flex-col md:flex-row justify-between text-sm gap-2">
          <p>© 2026 Elastic Tree Marketing Services Pvt. Ltd.</p>
          <p>AI Gaze™ · Predictive Eye Tracking</p>
        </div>
      </div>
    </footer>
  );
}
