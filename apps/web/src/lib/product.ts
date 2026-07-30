/** Public AI Gaze™ Streamlit studio URL (Railway) */
export const AI_GAZE_STUDIO_URL =
  process.env.NEXT_PUBLIC_AI_GAZE_STUDIO_URL ?? "https://aigaze-production.up.railway.app";

export const PRICING_PLANS = [
  {
    id: "starter",
    name: "Starter",
    priceLabel: "₹2,999",
    period: "/ month",
    blurb: "For freelancers and small teams testing a few creatives each month.",
    features: [
      "20 analyses / month",
      "Heat map, hot spot, gaze path",
      "Clarity & top elements",
      "PDF report export",
      "Email support",
    ],
    featured: false,
    cta: "Launch Studio",
  },
  {
    id: "growth",
    name: "Growth",
    priceLabel: "₹7,999",
    period: "/ month",
    blurb: "For brands and agencies running regular creative & pack QA.",
    features: [
      "80 analyses / month",
      "Everything in Starter",
      "A/B variant compare",
      "Face pull & attention balance",
      "Priority support · shared seats (3)",
    ],
    featured: true,
    cta: "Choose Growth",
  },
  {
    id: "enterprise",
    name: "Enterprise",
    priceLabel: "Custom",
    period: "from ₹19,999/mo",
    blurb: "For multi-brand teams needing volume, SLAs, and white-label.",
    features: [
      "Unlimited / high-volume credits",
      "Team seats & SSO (on request)",
      "API / batch workflow options",
      "White-label PDF branding",
      "Dedicated Elastic Tree researcher",
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
