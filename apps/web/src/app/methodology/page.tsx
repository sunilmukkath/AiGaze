import Link from "next/link";
import { ETHeader } from "@/components/et/ETHeader";
import { ETFooter } from "@/components/et/ETFooter";
import { AI_GAZE_STUDIO_URL } from "@/lib/product";

export default function MethodologyPage() {
  return (
    <>
      <ETHeader />
      <div className="min-h-screen pt-24 pb-20 px-6">
        <div className="max-w-3xl mx-auto">
          <p className="et-section-label mb-3">Method</p>
          <h1 className="et-display text-4xl font-black text-[var(--text-primary)] mb-6 tracking-tight">
            How AI Gaze™ predicts attention
          </h1>
          <div className="space-y-6 text-[var(--text-secondary)] leading-relaxed">
            <p>
              AI Gaze™ is Elastic Tree&apos;s predictive eye-tracking platform. It simulates pre-attentive vision —
              what people notice in the first ~3 seconds — without lab hardware.
            </p>
            <p>
              The model draws on visual drivers known to capture attention: edges, colour contrast (R/G, B/Y),
              intensity, and faces. Outputs include heat maps, hot-spot tiers, gaze sequence, clarity scoring, and
              branded PDF reports for creative QA.
            </p>
            <p>
              Validated against real eye-tracking sessions with approximately <strong className="text-[var(--text-primary)]">92% accuracy</strong>, AI Gaze™
              is typically <strong className="text-[var(--text-primary)]">~60% lower cost</strong> than traditional eye-tracking studies, with remote setup
              and fast turnaround.
            </p>
            <div className="et-card p-6 mt-8">
              <h2 className="text-[var(--text-primary)] font-semibold mb-2">Try it in the studio</h2>
              <p className="text-sm mb-4">
                Upload a creative and explore heat map, hot spot, gaze path, and export in one workspace.
              </p>
              <a href={AI_GAZE_STUDIO_URL} target="_blank" rel="noopener noreferrer" className="et-btn-primary text-sm inline-flex">
                Launch Studio
              </a>
            </div>
            <p className="text-sm pt-4">
              <Link href="/" className="text-[var(--amber)] hover:underline">
                ← Back to AI Gaze
              </Link>
            </p>
          </div>
        </div>
      </div>
      <ETFooter />
    </>
  );
}
