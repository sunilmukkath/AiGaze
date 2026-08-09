import Link from "next/link";
import { ETHeader } from "@/components/et/ETHeader";
import { ETFooter } from "@/components/et/ETFooter";
import { AI_GAZE_STUDIO_URL, PRICING_PLANS } from "@/lib/product";

export default function PricingPage() {
  return (
    <>
      <ETHeader />
      <div className="min-h-screen pt-24 pb-20 px-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="et-display text-4xl font-black text-[var(--text-primary)] text-center mb-4">
            Pricing
          </h1>
          <p className="text-[var(--text-secondary)] text-center mb-12">
            Per-creative and pack pricing · All prices exclusive of GST
          </p>
          <div className="grid md:grid-cols-3 gap-6">
            {PRICING_PLANS.map((plan) => (
              <div
                key={plan.id}
                className={`et-card p-6 flex flex-col ${
                  plan.featured ? "border-[var(--teal)] ring-1 ring-[var(--teal)]/30" : ""
                }`}
              >
                <h3 className="text-lg font-semibold text-[var(--text-primary)]">{plan.name}</h3>
                <p className="text-3xl font-bold text-[var(--teal)] my-4">
                  {plan.priceLabel}
                  <span className="text-sm font-medium text-[var(--text-muted)] ml-1">{plan.period}</span>
                </p>
                <p className="text-sm text-[var(--text-secondary)] mb-4">{plan.blurb}</p>
                <ul className="space-y-2 text-sm text-[var(--text-secondary)] flex-1 mb-6">
                  {plan.features.map((f) => (
                    <li key={f} className="flex gap-2">
                      <span className="text-[var(--teal)] font-bold">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>
                <a
                  href={`mailto:sunilmukkath@elastictree.com?subject=AI%20Gaze%20${encodeURIComponent(plan.name)}`}
                  className="et-btn-primary w-full text-sm justify-center"
                >
                  {plan.cta}
                </a>
              </div>
            ))}
          </div>
          <p className="text-center text-[var(--text-secondary)] text-sm mt-8">
            Prices in INR, exclusive of GST · Studio demo:{" "}
            <a href={AI_GAZE_STUDIO_URL} className="text-[var(--amber)] font-medium" target="_blank" rel="noreferrer">
              Launch Studio
            </a>
            {" · "}
            <Link href="mailto:sunilmukkath@elastictree.com" className="text-[var(--amber)] font-medium">
              Contact Elastic Tree
            </Link>
          </p>
        </div>
      </div>
      <ETFooter />
    </>
  );
}
