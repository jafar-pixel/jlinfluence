# JL — Master Project & Platform Charter
Updated: 2026-11-15

## One family. One umbrella. Distinct products.

**JL Influence is Jafar and Liz.** It is their story, values, work, impact, and the home of the platforms they create. JL Influence is **not** an athlete-training landing page, nor is it a synonym for PROVEN.

**Proposed core message:** “JL Influence — The influence we make. The work that proves it.”

**Jafar:** coach, athlete developer, builder, inventor and media/software platform creator. His sports-performance knowledge and human-development method have a dedicated product home in PROVEN.

**Liz:** an accomplished therapist and a central co-founder voice, not merely a supporting sports-mindset feature. Her professional work, human impact, and bright-light identity deserve their own fully developed story. Do not depict her as working at Rainier Springs; verify her current public practice and professional presentation before publication. Do not imply that Liz personally administers sport-specific tests unless separately confirmed.

The founders are people; the products are manifestations of their work.

## Brand architecture and destination roles

| Destination | Role | What belongs there |
|---|---|---|
| **JL Influence** — `jlinfluence.com` | Parent identity, Jafar + Liz, mission, impact, portfolio, contact | Founder stories, values, work, community and the ecosystem of ventures |
| **PROVEN** — `proven.jl-stream.com` | Standalone sports-performance product | Benchmark Method, speed assessments, training, coach and athlete experience, exercise and video engines, performance tracking and training services |
| **JL MediaVision** | Independent media technology product | Personalized streaming, TV apps, sports hub, device experiences |
| **Black Star Athletic Academy** | Distinct athletic development organization/program | Its own identity, athletic academy history, training offerings and relationship to JL Speed/PROVEN |
| **JL Vision, JL Speed, JL Pulse, MovementOS** | Methods, technologies and constituent products/features, subject to final naming | Clarify how each supports PROVEN without creating duplicate applications |
| **Liz's clinical work** | Separate professional service | Accurate therapy experience, philosophy, qualifications and links, subject to Liz's approval; not treated as a PROVEN feature |

Do not erase legacy brand names, logos, assets, or routes. Map and redirect only after validating what each route currently serves.

## Working site IA — JL Influence

1. Home: Jafar and Liz together, “Our influence is our work.”
2. Our Story: two individual stories and their shared journey
3. Our Work: projects, ventures and measurable impact
4. Platforms: PROVEN, JL MediaVision, Black Star Athletic Academy, and verified emerging ventures
5. Community / Impact: authentic activities, people and outcomes, not invented testimonials
6. Connect: separate personal/professional contact pathways where appropriate

Visual: emotionally warm, premium, human, distinctive dual-founder identity. Founder photographs and supplied JL brand marks must be used unchanged; do not fabricate their likeness or logos. Keep existing design tokens until originals are verified.

## Working site IA — PROVEN

1. Home: PROVEN as the sports performance and benchmark platform
2. The Benchmark Method: assess → compare → prescribe → train → reassess → adjust
3. Training: JL Speed and Black Star context where applicable, coaching options, drills
4. For Athletes & Parents: athlete journey, test results, training and progress
5. For Coaches / Teams: assessments, libraries, recordings, AI-assisted analysis, dashboard
6. Technology: JL Vision, supported integrations, planned sensor capabilities clearly labeled
7. Sign in: coach and athlete workspace, no cross-routing to the JL Influence marketing homepage

Use `proven.jl-stream.com` as the immediate secured origin. Keep existing auth, Supabase, and GitHub development branch; do not assume it has been deployed with the latest code.

## Source-of-truth and repository inventory

GitHub sources discovered through the connected account:
- `jafar-pixel/jl-pulse` — active PROVEN + legacy JL Influence Next.js site code; PROVEN PR #60 on `feature/proven-exercise-library-phase1` still draft
- `jafar-pixel/jlinfluence` — distinct existing repo, currently lightweight; use `docs/JL_MASTER_PROJECT.md` here as the **JL cross-product planning index**, not as an automatic production site replacement
- `jafar-pixel/harbor` — JL MediaVision development
- `jafar-pixel/Fairway-Studio` — additional app, ownership and relationship to JL portfolio to verify
- Other accessible repositories: `next-video-starter`, `jl-next-video`, `website-clone`, `jafar-career`, `dispatchadispatcharr_ranked_matchups-jl-networkrr_ranked_matchups`, `jl-netfin-riven-frontend`, `jl-netfin-documents`, `jl-netfin-iptv-streaming-frontend`, `jl-network-player`

A repository index is not the same as consolidating code. Preserve independent release, security and data boundaries.

## Chat and asset collection

Build a central source registry with fields:
- Product, original chat/project title, original URL/reference, date, decisions, latest status, asset links, code repo/branch, destination site, implementation vs concept, pending review.
- Prior conversation families to index: JL Influence positioning/branding, PROVEN mockup and exercise library, “Send All URLs”, Claude review, Benchmark Method implementation, JL Vision, JL Speed, JL Pulse/MovementOS, JL MediaVision/Harbor, JL Network/Netfin, Black Star Athletic Academy, Liz's professional presentation, logos/brand board, Fairway.
- Source authority: verified current production behavior > committed code > uploaded project source > recorded decision > speculative design idea.
- Separate legacy decisions from current direction, recording superseded site hosting and naming.

**Important limitation:** GitHub tools cannot move ChatGPT conversations or create/reparent ChatGPT Projects. The actual ChatGPT umbrella Project named `JL` must be created in ChatGPT; existing chats need to be moved/linked through the ChatGPT interface or provided as exports. No claim that all historical chats have already been imported.

## Migration sequence (no silent cutovers)

1. Collect authentic JL logo/brand assets, Jafar/Liz portraits and approved professional bios; audit existing pages and routes.
2. Create a content migration map: current JL Influence athletics pages → new PROVEN sections; preserve functioning app routes and backend.
3. Design JL Influence around both founders and the venture portfolio.
4. Design PROVEN as a fully independent product site while retaining its real engine and protected dashboard.
5. Validate links, SEO redirects, auth, video playback, coach/athlete privileges, Cloudflare Access and mobile.
6. Deploy each site independently with tested rollback; no automatic repo merge or destructive rename.

## Guardrails

- No invented identity assets, testimonials, athlete results, qualifications or live AI capabilities.
- Never describe clinical therapy as a feature of sports coaching without Liz's approval and correct professional context.
- Never make JL Influence just another PROVEN landing page.
- Preserve the previously agreed dark/animated visual assets unless replaced by an approved design.
- Protect athlete and patient information; separate auth/access and data roles appropriately.
