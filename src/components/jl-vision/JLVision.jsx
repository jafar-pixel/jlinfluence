/**
 * JLVision.jsx
 * Converted from the "JL Vision — See the Feedback" landing-page artifact.
 *
 * A self-contained marketing/product page for JL Vision — the "See" layer of
 * the JL Influence Movement OS. Records a clip → AR coaching review + drill plan.
 *
 * Everything is scoped under the .jlvision root (see JLVision.css). The three
 * inline scripts from the artifact are ported to hooks:
 *   - theme toggle          → useState + data-theme on the root
 *   - reveal-on-scroll (.rv) → IntersectionObserver in useEffect
 *   - hero particle network  → canvas animation in useEffect
 *
 * Binary assets (font, athlete image, demo clip) ship alongside as real files
 * and are resolved by the bundler.
 */

import React, { useEffect, useRef, useState } from 'react';
import './JLVision.css';
import athlete from './assets/athlete.webp';
import demoVideo from './assets/demo.mp4';

// ─── Reusable bits ───────────────────────────────────────────────────────────

// The JL Vision iris mark. `grad` must be unique per instance so multiple marks
// on the page don't share a gradient id.
function IrisMark({ className, grad, style, full = true }) {
  return (
    <svg className={className} style={style} viewBox="0 0 100 100" aria-hidden="true">
      <defs>
        <radialGradient id={grad} cx="42%" cy="36%" r="72%">
          <stop offset="0" stopColor="#8fe3ff" />
          <stop offset="1" stopColor="#0a63ff" />
        </radialGradient>
      </defs>
      <path d="M50 6 A44 44 0 0 0 50 94" fill="none" stroke="#00e5ff" strokeWidth="4" strokeLinecap="round" />
      <path d="M50 6 A44 44 0 0 1 50 94" fill="none" stroke="#ff7a00" strokeWidth="4" strokeLinecap="round" />
      {full && (
        <>
          <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(150,170,190,.4)" strokeWidth="4" strokeDasharray="1 6.5" />
          <circle cx="50" cy="50" r="33" fill="none" stroke="rgba(150,170,190,.22)" strokeWidth="1" />
        </>
      )}
      <path
        d="M20 50 Q50 27 80 50 Q50 73 20 50 Z"
        fill={full ? 'rgba(8,18,28,.3)' : 'none'}
        stroke="#dbe6f0"
        strokeWidth={full ? '3.4' : '3.6'}
        strokeLinejoin="round"
      />
      <circle cx="50" cy="50" r="10.5" fill={`url(#${grad})`} />
      {full && <circle cx="45.5" cy="45.5" r="3" fill="#eafaff" />}
    </svg>
  );
}

const HERO_STATS = [
  { b: 'Seconds', s: 'clip → review' },
  { b: 'On phone', s: 'iOS · Android' },
  { b: '30 yrs', s: 'JL coaching' },
  { b: 'Any sport', s: 'track · ball · youth' },
];

const STEPS = [
  { no: '01', g: '◉', h: 'Record or upload', p: 'Film on your phone, or pull from Photos, Files, or Downloads. Clips save straight to your library.' },
  { no: '02', g: '✦', h: 'AI sees it', p: 'Pose analysis + JL coaching logic read the clip against your profile — the limiter, the phase, the cue.' },
  { no: '03', g: '▤', h: 'Review + program', p: 'Get an AR coaching video and drills prescribed from your exerciseDB. Your coach is notified to review.' },
];

const LOOP = [
  { b: 'Feel', span: 'Proprioceptive & haptic cues build internal awareness.', on: false },
  { b: 'See', span: 'JL Vision makes sprint mechanics, joint angles, and limiters visible.', on: true },
  { b: 'Learn', span: 'Insight becomes a cue, a drill, and a plan the athlete understands.', on: false },
  { b: 'Repeat', span: 'Every rep feeds the next — the loop tightens, the athlete gets faster.', on: false },
];

const METRICS = [
  { k: 'HIP L', v: '54°', tone: 'good', words: 'Good projection' },
  { k: 'KNEE L', v: '121°', tone: 'risk', words: 'Needs cleaner shin angle' },
  { k: 'KNEE R', v: '118°', tone: 'risk', words: 'Improve front-side recovery' },
  { k: 'ANKLE L', v: '19°', tone: 'opt', words: 'Efficient dorsiflexion' },
  { k: 'ANKLE R', v: '17°', tone: 'opt', words: 'Strong ground stiffness' },
];

const CREDITS = [
  { n: '20', s: 'credits', p: '$9.99', free: false },
  { n: '60', s: 'credits · best value', p: '$24.99', free: false },
  { n: '150', s: 'credits', p: '$49.99', free: false },
  { n: '0', s: 'coach review', p: 'Always free', free: true },
];

const ECO = [
  { k: 'OS', b: 'MovementOS', s: 'Neural router connecting every layer', href: 'https://jlinfluence.com', ext: true, here: false },
  { k: 'SEE', b: 'JL Vision', s: 'Video evidence & AR pose analysis', href: '#top', ext: false, here: true },
  { k: 'FEEL', b: 'JL Pulse', s: 'Haptic feedback & body awareness', href: 'https://jlinfluence.com', ext: true, here: false },
  { k: 'TRAIN', b: 'JL Speed', s: 'The real training floor', href: 'https://jlinfluence.com', ext: true, here: false },
];

const EXT = { target: '_blank', rel: 'noopener' };

// ─── Component ───────────────────────────────────────────────────────────────

export default function JLVision() {
  // theme: null = follow OS; 'light' | 'dark' = forced
  const [theme, setTheme] = useState(null);
  const rootRef = useRef(null);
  const particlesRef = useRef(null);

  const toggleTheme = () => {
    setTheme((cur) => {
      const resolved = cur || (window.matchMedia('(prefers-color-scheme:light)').matches ? 'light' : 'dark');
      return resolved === 'dark' ? 'light' : 'dark';
    });
  };

  // reveal-on-scroll for .rv elements
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const els = root.querySelectorAll('.rv');
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        }),
      { threshold: 0.14 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  // hero particle network
  useEffect(() => {
    const c = particlesRef.current;
    if (!c) return;
    const g = c.getContext('2d');
    let W, H, pts, raf;

    const size = () => {
      const r = c.parentElement.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = r.width;
      H = r.height;
      c.width = W * dpr;
      c.height = H * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const init = () => {
      size();
      const n = Math.min(56, Math.round(W / 22));
      pts = Array.from({ length: n }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
      }));
    };
    const step = () => {
      g.clearRect(0, 0, W, H);
      for (const p of pts) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i];
          const b = pts[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < 118) {
            g.strokeStyle =
              (i + j) % 2
                ? 'rgba(0,229,255,' + 0.16 * (1 - d / 118) + ')'
                : 'rgba(255,122,0,' + 0.13 * (1 - d / 118) + ')';
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(a.x, a.y);
            g.lineTo(b.x, b.y);
            g.stroke();
          }
        }
      }
      for (const p of pts) {
        g.fillStyle = 'rgba(0,229,255,.6)';
        g.beginPath();
        g.arc(p.x, p.y, 1.5, 0, 7);
        g.fill();
      }
      raf = requestAnimationFrame(step);
    };
    const onResize = () => {
      cancelAnimationFrame(raf);
      init();
      step();
    };

    init();
    step();
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  const videoProps = { src: demoVideo, autoPlay: true, loop: true, muted: true, playsInline: true };

  return (
    <div className="jlvision" ref={rootRef} data-theme={theme || undefined}>
      <nav>
        <div className="wrap navin">
          <a className="logo" href="#top">
            <IrisMark className="mk" grad="jlv-irisA" />
            <b>JL<span> Vision</span></b>
          </a>
          <div className="navlinks">
            <a href="#how">How it works</a>
            <a href="#loop">The loop</a>
            <a href="#sides">Athletes &amp; coaches</a>
            <a href="#pricing">Pricing</a>
          </div>
          <div className="navsp" />
          <button className="tgl" onClick={toggleTheme} aria-label="Toggle theme">◐</button>
          <a className="ext" href="https://jlinfluence.com" {...EXT}>jlinfluence.com ↗</a>
          <a className="btn primary" href="https://jlinfluence.com" {...EXT} style={{ padding: '9px 16px' }}>Start</a>
        </div>
      </nav>
      <div id="top" />

      <header className="hero">
        <canvas id="particles" ref={particlesRef} />
        <div className="blob o" />
        <div className="blob c" />
        <div className="wrap herogrid">
          <div className="hcopy">
            <span className="eyebrow">SEE · the JL Influence vision system</span>
            <h1>See the<br />
              <em>feedback.</em></h1>
            <p className="sub">
              Record a clip on your phone. JL&nbsp;Vision turns it into an{' '}
              <b style={{ color: 'var(--text)' }}>AR coaching review</b> and a drill plan — matched to your sport, in seconds.
            </p>
            <div className="hactions">
              <a className="btn primary" href="https://jlinfluence.com" {...EXT}>▶ Start a free review</a>
              <a className="btn glass" href="#surface">See a review</a>
            </div>
            <div className="stats">
              {HERO_STATS.map((s) => (
                <div className="stat" key={s.b}>
                  <b>{s.b}</b>
                  <small>{s.s}</small>
                </div>
              ))}
            </div>
          </div>
          <div className="hstage">
            <div className="glowdisc" />
            <img
              className="hathlete"
              src={athlete}
              alt="Sprinter in stride with an AR pose overlay tracking joints"
              width="620"
              height="778"
            />
            <span className="archip see"><b>SEE</b> · pose tracked</span>
            <span className="archip knee"><span className="k">drive</span> <b>knee ↑</b></span>
            <div className="appphone">
              <div className="fr">
                <div className="appscr">
                  <div className="nub" />
                  <div className="aptop"><span className="d" /><b>JL Vision</b><small>0:07</small></div>
                  <div className="apvid">
                    <video {...videoProps} />
                    <span className="aptag">WICKET · REP 2</span>
                    <div className="ovc"><i>LIMITER · ARMS</i><br />Cheek to pocket — don't cross the zipper.</div>
                  </div>
                  <div className="apbar"><span className="pl">▶</span><span className="sc"><i /></span><small>0.25×</small></div>
                  <div className="apdrill">✦ 4 drills prescribed · coach notified</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section id="how">
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">How it works</span>
            <h2>Clip in. Coaching out.</h2>
            <p>No dashboards to learn. Add a video and your profile does the rest — the analysis already knows your sport, event, and goals.</p>
          </div>
          <div className="steps">
            {STEPS.map((s) => (
              <div className="step glass rv" key={s.no}>
                <div className="no">{s.no}</div>
                <div className="g">{s.g}</div>
                <h3>{s.h}</h3>
                <p>{s.p}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section
        id="loop"
        style={{ background: 'linear-gradient(180deg,transparent,color-mix(in srgb,var(--cyan) 4%,transparent),transparent)' }}
      >
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">The JL methodology</span>
            <h2>JL Vision is the <span style={{ color: 'var(--cyan)' }}>See</span>.</h2>
            <p>A closed loop builds a faster athlete: Feel → See → Learn → Repeat. A signal fires from the center and travels the loop — the tighter it runs, the faster the athlete develops.</p>
          </div>
          <div className="loopwrap">
            <div className="loopviz rv">
              <div className="loopring spin" />
              <div className="signal a" />
              <div className="signal b" />
              <div className="node feel glass"><b>Feel</b><small>PULSE</small></div>
              <div className="node see glass"><b>See</b><small>VISION</small></div>
              <div className="node learn glass"><b>Learn</b><small>COACH</small></div>
              <div className="node repeat glass"><b>Repeat</b><small>SPEED</small></div>
              <div className="core"><span className="r" /><span className="r" /><span className="r" /><i /></div>
            </div>
            <ul className="rv">
              {LOOP.map((l) => (
                <li className={l.on ? 'on' : undefined} key={l.b}>
                  <b>{l.b}</b>
                  <span>{l.span}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section id="surface">
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">The surface</span>
            <h2>AR you can coach from.</h2>
            <p>Overlays drawn from real pose analysis — skeleton, the arm crossing the midline, and{' '}
              <b style={{ color: 'var(--text)' }}>ground-contact zones</b> under the foot. Honest by default: a visual review now, verified metrics when pose, calibration, and IMU sync in.</p>
          </div>
          <div className="glass glasspanel rv">
            <div className="showgrid">
              <div className="vstage">
                <video {...videoProps} className="vfill" />
                <div className="chip see"><b>AR REVIEW · WICKET</b></div>
                <div className="cueb">
                  <span className="cx">Ground contact</span>
                  <b>Attack the ground under the hips — step over, strike down.</b>
                </div>
              </div>
              <div style={{ padding: '6px 8px' }}>
                <div className="mono" style={{ fontSize: '11px', letterSpacing: '.14em', color: 'var(--faint)' }}>CALEB · SPRINTS · U16</div>
                <h3 style={{ fontSize: '16px', margin: '8px 0 10px' }}>Primary limiter — arms &amp; torso</h3>
                <p style={{ color: 'var(--muted)', fontSize: '13.5px', margin: 0 }}>The arm crosses the midline, driving side-to-side rotation, so force leaks sideways as spacing opens. Rhythm and forward intent are strengths.</p>
                <div style={{ marginTop: '16px', display: 'flex', flexWrap: 'wrap', gap: '7px' }}>
                  <span className="chip" style={{ position: 'static' }}>▶ Seated arm drives</span>
                  <span className="chip" style={{ position: 'static' }}>▶ Wall arm switches</span>
                  <span className="chip" style={{ position: 'static' }}>▶ Ankle pogos</span>
                </div>
                <div className="mono" style={{ fontSize: '10.5px', color: 'var(--cyan)', marginTop: '14px' }}>✦ 4 drills prescribed · coach notified</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="coach">
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">Coach console</span>
            <h2>The dashboard coaches live in.</h2>
            <p>On tablet or desktop: the clip, biomechanical angles in{' '}
              <b style={{ color: 'var(--text)' }}>plain language</b>, and coach-to-athlete notes. Verified metrics unlock when pose, calibration &amp; IMU sync.</p>
          </div>
          <div className="tabwrap rv">
            <div className="tabdev">
              <div className="tabscr">
                <div className="tabtop">
                  <IrisMark className="tmk" grad="jlv-irisC" full={false} />
                  <b className="disp">JL Vision</b>
                  <span className="tsub">Video caption dashboard</span>
                  <span className="tsp" />
                  <span className="tath">
                    <span className="tav">M</span>
                    <span><b>Marcus T.</b><small>Age 18 · Varsity sprinter</small></span>
                  </span>
                  <span className="tframe">FRAME 237 / 360</span>
                </div>
                <div className="tabmain">
                  <div className="tvid">
                    <video {...videoProps} className="vfill" />
                    <div className="tcallout a" style={{ top: '16%', left: '12%' }}>◎ Project forward</div>
                    <div className="tcallout a" style={{ top: '34%', right: '8%' }}>☺ Drive knee through</div>
                    <div className="tcallout o" style={{ bottom: '18%', right: '16%' }}>Push back, not up</div>
                  </div>
                  <div className="tmetrics">
                    {METRICS.map((m) => (
                      <div className="mrow" key={m.k}>
                        <b>{m.k}</b>
                        <span className={`ga ${m.tone}`}>{m.v}</span>
                        <span className={`words ${m.tone}`}>{m.words}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="tabnotes">
                  <b>Coach to athlete —</b> Maintain projection through the first 3 steps · improve front-side knee recovery · strike under center of mass. <span style={{ color: 'var(--cyan)' }}>Stay powerful, stay relaxed.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="sides">
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">Two sides, one loop</span>
            <h2>Built for athletes &amp; coaches.</h2>
          </div>
          <div className="sides">
            <div className="side ath glass rv">
              <span className="tag">For athletes</span>
              <h3>Train with eyes on every rep.</h3>
              <ul>
                <li>Profile sets your sport, event, level &amp; goals — it's your prompt</li>
                <li>Record or upload; a saved video library</li>
                <li>AR review + drills prescribed to your sport</li>
                <li>Invite a coach review — always free</li>
                <li>Credits fund AI analysis &amp; program upgrades</li>
              </ul>
            </div>
            <div className="side co glass rv">
              <span className="tag">For coaches</span>
              <h3>Coach more athletes, on tablet or desktop.</h3>
              <ul>
                <li>Roster of every athlete and what they're working on</li>
                <li>Get notified when AI prescribes — approve or adjust</li>
                <li>Assign drills from Speed &amp; Agility, S&amp;C, sport-specific</li>
                <li>Review clips and progress in one console</li>
                <li>Your judgment stays in the loop</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section
        id="pricing"
        style={{ background: 'linear-gradient(180deg,transparent,color-mix(in srgb,var(--orange) 5%,transparent),transparent)' }}
      >
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">Simple credits</span>
            <h2>Pay for AI. Coaching stays human.</h2>
            <p>Credits fund the AI analysis and program upgrades. A human coach review always costs zero.</p>
          </div>
          <div className="credits">
            {CREDITS.map((c) => (
              <div className={`pc glass rv${c.free ? ' free' : ''}`} key={c.s}>
                <div className="n">{c.n}</div>
                <small>{c.s}</small>
                <div className="p">{c.p}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="eco">
        <div className="wrap">
          <div className="shd rv">
            <span className="eyebrow">Part of the movement</span>
            <h2>One system. Four layers.</h2>
            <p>JL Vision is the <span style={{ color: 'var(--orange)' }}>See</span> layer of the JL Influence Movement OS.</p>
          </div>
          <div className="eco">
            {ECO.map((e) => (
              <a
                className={`ecoc glass rv${e.here ? ' here' : ''}`}
                href={e.href}
                key={e.b}
                {...(e.ext ? EXT : {})}
              >
                <div className="k">{e.k}</div>
                <b>{e.b}</b>
                <small>{e.s}</small>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section>
        <div className="wrap">
          <div className="final glass rv">
            <div className="blob o" style={{ opacity: 0.3, top: '-160px', left: '20%' }} />
            <div className="blob c" style={{ opacity: 0.3, bottom: '-200px', right: '10%' }} />
            <div style={{ position: 'relative', zIndex: 2 }}>
              <span className="eyebrow">Start training</span>
              <h2 style={{ marginTop: '12px' }}>Build speed. See it happen.</h2>
              <p>Train with Coach Jafar and the JL Influence system — backed by science, biomechanics, and 30 years of coaching.</p>
              <div className="hactions">
                <a className="btn primary" href="https://jlinfluence.com" {...EXT}>Book athlete training</a>
                <a className="btn glass" href="https://jlinfluence.com" {...EXT}>Explore JL Influence ↗</a>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap fin">
          <a className="logo" href="https://jlinfluence.com" {...EXT} style={{ gap: '9px' }}>
            <IrisMark className="mk" grad="jlv-irisB" full={false} style={{ width: '22px', height: '22px' }} />
            <b className="disp" style={{ fontSize: '13px' }}>JL Influence</b>
          </a>
          <div className="navsp" />
          <span>© 2026 JL Influence · The Training Movement OS</span>
          <a href="https://jlinfluence.com" {...EXT}>jlinfluence.com ↗</a>
        </div>
      </footer>
    </div>
  );
}
