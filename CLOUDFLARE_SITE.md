# JL Influence — Cloudflare Website

This repository contains several historic JL components, **but its Cloudflare production website is the independent founder-led page in `public/index.html`**.

## Hosting

- Cloudflare Worker name: `jlinfluence`
- Worker domain: `jlinfluence.jafar-354.workers.dev`
- Production domain (after review): `jlinfluence.com` and `www.jlinfluence.com`
- Config: root `wrangler.jsonc`
- Deploy command: `npx wrangler deploy`
- Build command: none (static HTML/CSS; no npm build required)
- Root directory: `/`

Wrangler's `assets.directory` points explicitly at `./public`, **not** the separate JL Vision Vite app in `jl-vision/`.

## Content scope

JL Influence = Jafar + Liz, their story, shared impact, and platform family. PROVEN and JL MediaVision are separate products. Only PROVEN has a live outbound link until the other product URLs have been verified.

The first page is a real deployable responsive HTML site, but it intentionally uses typographic initials rather than fabricated founders' photographs or recreated JL logos. Replace with approved original brand assets and portraits when supplied.

## Deployment steps

1. Review branch `feature/jl-influence-cloudflare-home` and merge when approved.
2. In the existing Cloudflare Workers `jlinfluence` project: build command `None`; deploy command `npx wrangler deploy`; root `/`.
3. Trigger Git deployment for main and inspect build logs and response at `https://jlinfluence.jafar-354.workers.dev`.
4. Confirm title is `JL Influence — Jafar & Liz`, hero, all navigation anchors, and PROVEN link work on mobile/desktop.
5. Only after Cloudflare DNS is active and email records are verified, connect custom apex and www domains. Replace the former Vercel destination, keep MX/SPF/DKIM/DMARC and registrar independent.
6. Avoid adding Cloudflare Access to this public brand website unless deliberately desired; PROVEN's app access remains separate.

No existing JL Vision code or production DNS was changed as part of this draft branch.
