import Link from "next/link";
import { ArrowRight, CheckCircle2, Eye, Layers, Target, FileText } from "lucide-react";
import { ETHeader } from "@/components/et/ETHeader";
import { ETFooter } from "@/components/et/ETFooter";
import { AI_GAZE_STUDIO_URL, FEATURES, PRICING_PLANS } from "@/lib/product";

export default function HomePage() {
  return (
    <>
      <ETHeader />
      <main className="pt-16">
        {/* Hero */}
        <section className="et-hero relative px-6 pt-20 pb-24 overflow-hidden">
          <div className="et-hero-orb et-hero-orb--amber w-72 h-72 top-10 right-10 opacity-40 absolute blur-3xl pointer-events-none" aria-hidden />
          <div className="et-hero-orb et-hero-orb--cyan w-80 h-80 bottom-0 left-10 opacity-30 absolute blur-3xl pointer-events-none" aria-hidden />

          <div className="max-w-6xl mx-auto relative z-10">
            <div className="max-w-3xl">
              <span className="et-badge mb-6 inline-flex items-center gap-2">
                <Eye size={14} />
                <span className="et-brand-name text-xs tracking-wide">AI Gaze™</span>
              </span>
              <p className="et-section-label mb-4">Advanced Methods · Predictive Eye Tracking</p>
              <h1 className="et-display text-4xl md:text-[3.2rem] font-black leading-[1.08] tracking-tight mb-6 text-[var(--text-primary)]">
                See what gets attention in the{" "}
                <span className="et-gradient-text">first 3 seconds</span>.
              </h1>
              <p className="text-lg text-[var(--text-secondary)] leading-relaxed mb-8 max-w-xl">
                Upload packs, ads, and digital creatives — get heat maps, hot spots, gaze path, and a branded PDF
                report without hardware eye-tracking.
              </p>
              <div className="flex flex-wrap gap-4 mb-10">
                <a href={AI_GAZE_STUDIO_URL} target="_blank" rel="noopener noreferrer" className="et-btn-primary flex items-center gap-2">
                  Launch Studio <ArrowRight size={18} />
                </a>
                <Link href="/pricing" className="et-btn-secondary">
                  View Pricing
                </Link>
              </div>
              <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-[var(--text-muted)]">
                {["92% accuracy vs lab", "No hardware required", "~3s first-glance window", "–60% vs traditional ET"].map(
                  (t) => (
                    <span key={t} className="flex items-center gap-1.5">
                      <CheckCircle2 size={14} className="text-[var(--teal)]" /> {t}
                    </span>
                  ),
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Stats */}
        <section className="px-6 pb-16">
          <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { v: "92%", l: "Accuracy vs lab ET", c: "var(--amber)" },
              { v: "~3s", l: "First-glance window", c: "var(--teal)" },
              { v: "–60%", l: "Cost vs hardware", c: "var(--cyan)" },
              { v: "24h", l: "Typical turnaround", c: "#a78bfa" },
            ].map((s) => (
              <div key={s.l} className="et-card p-5 text-center">
                <p className="et-display text-2xl md:text-3xl font-black" style={{ color: s.c }}>
                  {s.v}
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-2 uppercase tracking-wide">{s.l}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Features */}
        <section id="features" className="px-6 py-20">
          <div className="max-w-6xl mx-auto">
            <p className="et-section-label mb-3 text-center">Platform</p>
            <h2 className="et-display text-3xl md:text-4xl font-black text-center text-[var(--text-primary)] mb-4 tracking-tight">
              Not a static heatmap — a living attention workspace
            </h2>
            <p className="text-center text-[var(--text-secondary)] max-w-2xl mx-auto mb-12">
              Everything your team needs to validate hierarchy before you print, shelf, or spend media.
            </p>
            <div className="grid md:grid-cols-2 gap-5">
              {FEATURES.map((f, i) => {
                const icons = [Target, Eye, Layers, FileText];
                const Icon = icons[i] ?? Eye;
                return (
                  <div key={f.title} className="et-card p-6 flex gap-4">
                    <div className="w-10 h-10 rounded-xl bg-[var(--amber)]/10 border border-[var(--amber)]/25 flex items-center justify-center shrink-0">
                      <Icon size={18} className="text-[var(--amber)]" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-[var(--text-primary)] mb-1.5">{f.title}</h3>
                      <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{f.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="px-6 py-20 border-t border-white/5">
          <div className="max-w-6xl mx-auto">
            <p className="et-section-label mb-3 text-center">How it works</p>
            <h2 className="et-display text-3xl font-black text-center text-[var(--text-primary)] mb-12 tracking-tight">
              Upload. Analyse. Decide.
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                { n: "01", t: "Upload creative", d: "Pack shot, shelf photo, ad, or digital layout — JPG/PNG/WebP." },
                { n: "02", t: "Run AI Gaze™", d: "Deep gaze prediction for heat map, hot spots, gaze path, and clarity." },
                { n: "03", t: "Export & act", d: "Share the branded PDF and fix hierarchy before go-live." },
              ].map((s) => (
                <div key={s.n} className="et-card p-6">
                  <p className="font-mono text-xs text-[var(--teal)] mb-3 tracking-widest">{s.n}</p>
                  <h3 className="font-semibold text-[var(--text-primary)] mb-2">{s.t}</h3>
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{s.d}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing teaser */}
        <section className="px-6 py-20 border-t border-white/5">
          <div className="max-w-6xl mx-auto">
            <p className="et-section-label mb-3 text-center">Pricing</p>
            <h2 className="et-display text-3xl font-black text-center text-[var(--text-primary)] mb-4 tracking-tight">
              Plans that fit creative QA
            </h2>
            <p className="text-center text-[var(--text-secondary)] mb-10">
              Transparent SaaS pricing · Custom studies via Elastic Tree
            </p>
            <div className="grid md:grid-cols-3 gap-5">
              {PRICING_PLANS.map((plan) => (
                <div
                  key={plan.id}
                  className={`et-card p-6 flex flex-col ${plan.featured ? "ring-1 ring-[var(--amber)]/50" : ""}`}
                >
                  <p className="text-xs font-mono uppercase tracking-widest text-[var(--teal)] mb-2">{plan.name}</p>
                  <p className="et-display text-3xl font-black text-[var(--text-primary)]">
                    {plan.priceLabel}
                    <span className="text-sm font-semibold text-[var(--text-muted)] ml-1">{plan.period}</span>
                  </p>
                  <p className="text-sm text-[var(--text-secondary)] mt-3 mb-5">{plan.blurb}</p>
                  <ul className="space-y-2 text-sm text-[var(--text-secondary)] flex-1 mb-6">
                    {plan.features.map((f) => (
                      <li key={f} className="flex gap-2">
                        <span className="text-[var(--teal)] font-bold">✓</span>
                        {f}
                      </li>
                    ))}
                  </ul>
                  {plan.id === "enterprise" ? (
                    <a href="mailto:sunil@elastictree.com" className="et-btn-secondary w-full justify-center text-sm">
                      {plan.cta}
                    </a>
                  ) : (
                    <a
                      href={AI_GAZE_STUDIO_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`${plan.featured ? "et-btn-primary" : "et-btn-secondary"} w-full justify-center text-sm`}
                    >
                      {plan.cta}
                    </a>
                  )}
                </div>
              ))}
            </div>
            <p className="text-center mt-8">
              <Link href="/pricing" className="text-[var(--amber)] text-sm font-medium hover:underline">
                Full pricing details →
              </Link>
            </p>
          </div>
        </section>

        {/* Studio CTA */}
        <section id="studio" className="px-6 py-20 border-t border-white/5">
          <div className="max-w-3xl mx-auto et-card p-10 text-center">
            <p className="et-section-label mb-3">Studio</p>
            <h2 className="et-display text-3xl font-black text-[var(--text-primary)] mb-4 tracking-tight">
              Ready to test a creative?
            </h2>
            <p className="text-[var(--text-secondary)] mb-8 leading-relaxed">
              Open the AI Gaze™ studio and run your first analysis in minutes.
            </p>
            <a href={AI_GAZE_STUDIO_URL} target="_blank" rel="noopener noreferrer" className="et-btn-primary inline-flex items-center gap-2">
              Launch Studio <ArrowRight size={18} />
            </a>
          </div>
        </section>
      </main>
      <ETFooter />
    </>
  );
}
