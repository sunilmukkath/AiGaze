/** Public AI Gaze™ Streamlit studio URL (Railway) */
export const AI_GAZE_STUDIO_URL =
  process.env.NEXT_PUBLIC_AI_GAZE_STUDIO_URL ?? "https://aigaze-production.up.railway.app";

export const PRICING_PLANS = [
  {
    id: "single",
    name: "Single Test",
    priceLabel: "₹4,500",
    period: "/ creative",
    blurb: "One creative asset — heatmap, clarity score, and PDF report.",
    features: [
      "1 creative asset",
      "Attention heatmap + clarity score",
      "PDF report",
    ],
    featured: false,
    cta: "Request quote",
  },
  {
    id: "pack10",
    name: "Pack of 10",
    priceLabel: "₹32,000",
    period: "(~₹3,200 ea.)",
    blurb: "Compare creatives across a set with shared benchmarking.",
    features: [
      "10 creative assets",
      "Comparative benchmarking across the set",
      "Heatmap + clarity for each asset",
      "PDF reports",
    ],
    featured: true,
    cta: "Request quote",
  },
  {
    id: "retainer",
    name: "Agency Retainer",
    priceLabel: "Custom",
    period: "quote",
    blurb: "Ongoing monthly volume with priority turnaround.",
    features: [
      "Ongoing monthly volume",
      "Priority turnaround",
      "Brand-tracking over time",
    ],
    featured: false,
    cta: "Talk to Sales",
  },
] as const;

export const FEATURES = [
  {
    title: "Heat Map & Hot Spots",
    desc: "See where first-glance attention lands — HIGH / MEDIUM / LOW zones on your creative.",
  },
  {
    title: "Gaze Path",
    desc: "Predicted fixation order within ~3 seconds so hierarchy matches intent.",
  },
  {
    title: "Clarity Score",
    desc: "Quantify how focused or fragmented attention is across the canvas.",
  },
  {
    title: "Branded PDF Report",
    desc: "Export an Elastic Tree–ready deck for client reviews and creative QA.",
  },
] as const;
