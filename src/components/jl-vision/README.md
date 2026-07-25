# JLVision

The **"JL Vision — See the Feedback"** landing page as a self-contained React component.
JL Vision is the *See* layer of the JL Influence Movement OS: record a clip → AR
coaching review + a drill plan matched to the athlete's sport.

Converted from the `JL Vision — See the Feedback` HTML artifact. All markup, the
Eurostile display font, the athlete image, and the demo clip travel with the
component — no external UI dependencies.

## Sections

- **Nav** — sticky, with theme toggle and the JL Vision iris mark
- **Hero** — particle-network canvas, animated blobs, athlete image + docked phone app
- **How it works** — 3-step clip → coaching flow
- **The loop** — orbital Feel → See → Learn → Repeat methodology diagram
- **The surface** — AR review panel (video + limiter analysis + prescribed drills)
- **Coach console** — tablet dashboard with plain-language biomechanics metrics
- **Athletes & coaches** — two-column value split
- **Pricing** — credit tiers (coach review always free)
- **Ecosystem** — the four MovementOS layers
- **Final CTA + footer**

## Visual style

- Dark-first palette (`--bg #07090d`) with orange `#ff7a00` + cyan `#00e5ff` accents
- Glassmorphism panels, animated gradient blobs, reveal-on-scroll
- Eurostile Extended display type (bundled subset) + monospace data type

## Usage

```jsx
import { JLVision } from './components/jl-vision';

<JLVision />
```

Render it full-bleed — it manages its own layout, background, and sticky nav:

```jsx
<div style={{ minHeight: '100vh' }}>
  <JLVision />
</div>
```

## Theming

The component follows the OS color scheme by default and exposes a toggle (`◐`)
that forces light/dark via `data-theme` on the `.jlvision` root. All styles are
scoped under `.jlvision`, and the keyframes are namespaced (`jlv-*`), so the
component is drop-in safe alongside other page content.

## Assets

Bundler-resolved files under `assets/`:

| File | Purpose |
|---|---|
| `eurostile-ext.otf` | Display font subset (referenced by `@font-face`) |
| `athlete.webp` | Hero sprinter image with AR overlay |
| `demo.mp4` | Looping clip used in the phone app, AR surface, and coach console |

## Build requirements

Assumes a bundler that handles CSS imports and static asset imports
(`.otf` / `.webp` / `.mp4`) — e.g. Vite, webpack, or Next.js — same as the other
components in this repo. No runtime dependencies beyond React.
