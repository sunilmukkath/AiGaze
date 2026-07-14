# AI Gaze™ Design System

Same Elastic Tree product chrome as Ethos Pulse — dark spectrum background, amber CTAs, teal/cyan accents. Product mark: `aigaze-logo.png` (header); Elastic Tree corporate mark in footer.

Aligned with [elastictree.com](https://www.elastictree.com).

## Color Palette (from ET site CSS)

| Token | Hex | Usage |
|-------|-----|-------|
| `--void` / `--et-navy` | `#0a1f4a` | Base background, theme-color |
| `--navy-deep` | `#090e2c` | Header/footer scrolled, sidebar |
| `--space` | `#0c2d5c` | Mid-tone surfaces |
| `--amber` / `--et-gold` | `#e8a820` | Primary CTA, nav active, stats |
| `--amber-light` | `#f5c842` | CTA gradient highlight |
| `--teal` / `--et-teal` | `#2dd4bf` | Brand accent, charts, links |
| `--cyan` | `#38bdf8` | Glows, borders, glass |
| `--text-primary` | `#f1f5f9` | Headings on dark |
| `--text-body` | `#e2e8f0` | Body copy |
| `--text-secondary` | `#cbd5e1` | Secondary copy |
| `--text-muted` | `#94a3b8` | Captions, labels |

## Typography

| Role | Font |
|------|------|
| Body | DM Sans |
| Buttons & stats | Outfit |
| Eyebrows / labels | DM Mono |

## Key Patterns

- **Background:** `.page-spectrum` — multi-stop gradient matching ET homepage
- **Primary button:** `.et-btn-primary` — amber gold pill gradient
- **Secondary button:** `.et-btn-secondary` — glass navy pill
- **Cards:** `.et-card` — space-card glass gradient with cyan border
- **Inputs:** `.et-input` — dark glass with amber border focus

## Implementation

Tokens live in `apps/web/src/app/globals.css`. Fonts loaded in `layout.tsx` via `next/font/google`.
