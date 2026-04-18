/**
 * BodyMap3D.jsx
 * Interactive 3D medical body map — React component
 * Matches reference: deep navy bg · cyan hex UI · ECG line · blue translucent anatomy
 *
 * Dependencies (add to your package.json):
 *   three@^0.162.0
 *   @react-three/fiber@^8.x
 *   @react-three/drei@^9.x
 *   @react-three/postprocessing@^2.x
 *
 * Usage:
 *   import BodyMap3D from './components/body-map/BodyMap3D';
 *   <BodyMap3D />
 */

import React, { useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF, Sphere, MeshDistortMaterial } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import './BodyMap3D.css';

// ─── Muscle Data ───────────────────────────────────────────────────────────────
const MUSCLES = {
  pectorals:   { name: 'Pectorals',        latin: 'Musculus pectoralis major',   group: 'Chest',    color: '#3b82f6', pos: [0, 0.55, 0.18],  size: 0.14 },
  deltoids:    { name: 'Deltoids',          latin: 'Musculus deltoideus',         group: 'Shoulder', color: '#6366f1', pos: [0.28, 0.62, 0.0], size: 0.1  },
  biceps:      { name: 'Biceps',            latin: 'Musculus biceps brachii',     group: 'Arms',     color: '#0ea5e9', pos: [0.32, 0.32, 0.06],size: 0.08 },
  triceps:     { name: 'Triceps',           latin: 'Musculus triceps brachii',    group: 'Arms',     color: '#22d3ee', pos: [0.32, 0.32,-0.06],size: 0.08 },
  trapezius:   { name: 'Trapezius',         latin: 'Musculus trapezius',          group: 'Back',     color: '#8b5cf6', pos: [0, 0.72,-0.16],   size: 0.14 },
  latissimus:  { name: 'Latissimus Dorsi',  latin: 'Musculus latissimus dorsi',   group: 'Back',     color: '#7c3aed', pos: [0.18, 0.38,-0.14],size: 0.13 },
  rectus:      { name: 'Rectus Abdominis',  latin: 'Musculus rectus abdominis',   group: 'Core',     color: '#10b981', pos: [0, 0.24, 0.16],   size: 0.11 },
  obliques:    { name: 'Obliques',          latin: 'Musculi obliqui abdominis',   group: 'Core',     color: '#059669', pos: [0.16, 0.22, 0.12],size: 0.09 },
  quadriceps:  { name: 'Quadriceps',        latin: 'Musculus quadriceps femoris', group: 'Legs',     color: '#f59e0b', pos: [0.13,-0.28, 0.13],size: 0.14 },
  hamstrings:  { name: 'Hamstrings',        latin: 'Musculi flexores genus',      group: 'Legs',     color: '#d97706', pos: [0.13,-0.28,-0.13],size: 0.14 },
  glutes:      { name: 'Glutes',            latin: 'Musculi glutei',              group: 'Legs',     color: '#ef4444', pos: [0,-0.08,-0.2],    size: 0.17 },
  calves:      { name: 'Calves',            latin: 'Musculi surales',             group: 'Legs',     color: '#fb923c', pos: [0.11,-0.62, 0.0], size: 0.09 },
  gastrocnemius:{ name: 'Gastrocnemius',   latin: 'Musculus gastrocnemius',      group: 'Legs',     color: '#f97316', pos: [0.11,-0.7,-0.04], size: 0.08 },
};

const MUSCLE_TIPS = {
  pectorals:   ['Full ROM bench press for max fiber recruitment', 'Include incline + decline for both heads', 'Cable flyes maintain constant tension'],
  deltoids:    ['Lateral raises isolate the medial head', 'Face pulls train the posterior head', 'Overhead press activates all 3 heads'],
  biceps:      ['Supinate at the top of every curl', 'Incline curls stretch the long head', 'Vary grip width to hit both heads'],
  triceps:     ['Overhead extensions place long head on full stretch', 'Close-grip bench allows heavy loading', 'Rope pushdowns flare out at the bottom'],
  trapezius:   ['Heavy shrugs build upper trap', 'Y-T-W exercises correct posture', 'Deadlifts provide systemic stimulus'],
  latissimus:  ['Full stretch at bottom of pull-ups is essential', 'Straight-arm pulldowns isolate the lat', 'Lean back on rows to hit lower insertion'],
  rectus:      ['Think ribs to pelvis — not hips to chest', 'Cable crunches allow progressive overload', 'Weighted carries build functional stability'],
  obliques:    ['Pallof press trains anti-rotation', 'Cable woodchops build rotational power', 'Side planks create high demand with low spine load'],
  quadriceps:  ['Forward knee travel maximizes quad recruitment', 'Low foot placement on leg press hits quads', 'Leg extensions isolate at terminal extension'],
  hamstrings:  ['RDLs provide hip hinge under load', 'Nordic curls build eccentric strength', 'Leg curls in hip flexion better isolate hams'],
  glutes:      ['Hip thrusts are superior for glute max at short position', 'Include both hip extension and abduction', 'Posterior pelvic tilt at top maximally contracts glute max'],
  calves:      ['Seated raises isolate the soleus', 'High reps OR heavy load — partial reps are useless', 'Full bottom stretch is the most important cue'],
  gastrocnemius: ['Standing calf raises target the gastrocnemius', 'Straight knee required for gastrocnemius isolation', 'Jump training provides plyometric stimulus'],
};

// ─── Body model built from anatomical proportions ─────────────────────────────
function AnatomicalBody({ selectedMuscle, hoveredMuscle, onMuscleClick, onMuscleHover }) {
  const groupRef = useRef();
  const clock = useRef(new THREE.Clock());

  useFrame(() => {
    if (groupRef.current) {
      const t = clock.current.getElapsedTime();
      groupRef.current.position.y = Math.sin(t * 0.5) * 0.015;
    }
  });

  const skinMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x1040a0),
    transparent: true,
    opacity: 0.62,
    roughness: 0.18,
    metalness: 0.0,
    transmission: 0.2,
    thickness: 0.5,
    emissive: new THREE.Color(0x0820a0),
    emissiveIntensity: 0.12,
    side: THREE.FrontSide,
  });

  const parts = [
    // Head
    { geo: <sphereGeometry args={[0.14, 32, 24]} />, pos: [0, 0.96, 0], scale: [1, 1.1, 0.95] },
    // Neck
    { geo: <cylinderGeometry args={[0.055, 0.07, 0.11, 20]} />, pos: [0, 0.79, 0] },
    // Chest
    { geo: <capsuleGeometry args={[0.19, 0.28, 16, 32]} />, pos: [0, 0.54, 0], scale: [1, 1, 0.75] },
    // Abdomen
    { geo: <capsuleGeometry args={[0.15, 0.18, 16, 32]} />, pos: [0, 0.27, 0], scale: [1, 1, 0.72] },
    // Pelvis
    { geo: <sphereGeometry args={[0.18, 24, 18]} />, pos: [0, 0.08, 0], scale: [1.05, 0.65, 0.85] },
    // Upper arms R/L
    { geo: <capsuleGeometry args={[0.05, 0.22, 12, 20]} />, pos: [0.3, 0.56, 0], rot: [0, 0, 0.15] },
    { geo: <capsuleGeometry args={[0.05, 0.22, 12, 20]} />, pos: [-0.3, 0.56, 0], rot: [0, 0, -0.15] },
    // Lower arms R/L
    { geo: <capsuleGeometry args={[0.038, 0.2, 12, 20]} />, pos: [0.35, 0.28, 0], rot: [0, 0, 0.18] },
    { geo: <capsuleGeometry args={[0.038, 0.2, 12, 20]} />, pos: [-0.35, 0.28, 0], rot: [0, 0, -0.18] },
    // Hands R/L
    { geo: <sphereGeometry args={[0.038, 14, 10]} />, pos: [0.4, 0.1, 0], scale: [1, 0.7, 0.6] },
    { geo: <sphereGeometry args={[0.038, 14, 10]} />, pos: [-0.4, 0.1, 0], scale: [1, 0.7, 0.6] },
    // Upper legs R/L
    { geo: <capsuleGeometry args={[0.08, 0.3, 14, 24]} />, pos: [0.12, -0.22, 0], rot: [0, 0, 0.04] },
    { geo: <capsuleGeometry args={[0.08, 0.3, 14, 24]} />, pos: [-0.12, -0.22, 0], rot: [0, 0, -0.04] },
    // Knees R/L
    { geo: <sphereGeometry args={[0.06, 16, 12]} />, pos: [0.12, -0.52, 0] },
    { geo: <sphereGeometry args={[0.06, 16, 12]} />, pos: [-0.12, -0.52, 0] },
    // Lower legs R/L
    { geo: <capsuleGeometry args={[0.05, 0.26, 12, 22]} />, pos: [0.11, -0.76, 0], rot: [0, 0, 0.02] },
    { geo: <capsuleGeometry args={[0.05, 0.26, 12, 22]} />, pos: [-0.11, -0.76, 0], rot: [0, 0, -0.02] },
    // Feet R/L
    { geo: <capsuleGeometry args={[0.035, 0.11, 10, 16]} />, pos: [0.11, -1.0, 0.05], rot: [Math.PI/2, 0, 0] },
    { geo: <capsuleGeometry args={[0.035, 0.11, 10, 16]} />, pos: [-0.11, -1.0, 0.05], rot: [Math.PI/2, 0, 0] },
  ];

  return (
    <group ref={groupRef}>
      {/* Skin body parts */}
      {parts.map((p, i) => (
        <mesh
          key={i}
          position={p.pos}
          rotation={p.rot || [0, 0, 0]}
          scale={p.scale || [1, 1, 1]}
          material={skinMat}
          castShadow
        >
          {p.geo}
        </mesh>
      ))}

      {/* Muscle zones */}
      {Object.entries(MUSCLES).map(([id, m]) => {
        const isSelected = selectedMuscle === id;
        const isHovered = hoveredMuscle === id;
        const active = isSelected || isHovered;
        const col = new THREE.Color(m.color);
        const bilateral = ['deltoids','biceps','triceps','latissimus','obliques','quadriceps','hamstrings','calves','gastrocnemius'];

        const muscleMat = new THREE.MeshPhysicalMaterial({
          color: col,
          transparent: true,
          opacity: active ? 0.88 : 0.0,
          roughness: 0.2,
          metalness: 0.05,
          emissive: col,
          emissiveIntensity: active ? (isSelected ? 0.55 : 0.25) : 0,
        });

        return (
          <React.Fragment key={id}>
            <mesh
              position={m.pos}
              material={muscleMat}
              onClick={(e) => { e.stopPropagation(); onMuscleClick(id); }}
              onPointerOver={(e) => { e.stopPropagation(); onMuscleHover(id); }}
              onPointerOut={() => onMuscleHover(null)}
            >
              <sphereGeometry args={[m.size, 18, 14]} />
            </mesh>
            {/* Mirror for bilateral */}
            {bilateral.includes(id) && (
              <mesh
                position={[-m.pos[0], m.pos[1], m.pos[2]]}
                material={muscleMat}
                onClick={(e) => { e.stopPropagation(); onMuscleClick(id); }}
                onPointerOver={(e) => { e.stopPropagation(); onMuscleHover(id); }}
                onPointerOut={() => onMuscleHover(null)}
              >
                <sphereGeometry args={[m.size, 18, 14]} />
              </mesh>
            )}
          </React.Fragment>
        );
      })}
    </group>
  );
}

// ─── ECG Line drawn on a canvas texture ────────────────────────────────────────
function ECGLine() {
  const meshRef = useRef();
  const texRef = useRef();
  const offset = useRef(0);

  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 64;
    texRef.current = new THREE.CanvasTexture(canvas);

    const update = () => {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 512, 64);
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 1.5;
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#00e5ff';
      ctx.beginPath();
      const amp = 22, mid = 32, spd = 0.06;
      for (let x = 0; x < 512; x++) {
        const t = (x / 512 + offset.current) * Math.PI * 10;
        let y = mid;
        const norm = ((x / 512 + offset.current) % 1);
        if (norm > 0.3 && norm < 0.32) y = mid - amp * 0.3;
        else if (norm > 0.32 && norm < 0.36) y = mid + amp * 2.5;
        else if (norm > 0.36 && norm < 0.40) y = mid - amp * 1.5;
        else if (norm > 0.40 && norm < 0.44) y = mid + amp * 0.5;
        else y = mid + Math.sin(t * 0.3) * 2;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      texRef.current.needsUpdate = true;
      offset.current += spd * 0.016;
    };

    let raf;
    const loop = () => { update(); raf = requestAnimationFrame(loop); };
    loop();
    return () => cancelAnimationFrame(raf);
  }, []);

  return null; // rendered as overlay in DOM instead
}

// ─── Hex grid corner decoration ────────────────────────────────────────────────
function HexCorner({ top, right, bottom, left }) {
  return (
    <svg
      className="bm-hex-corner"
      style={{ top, right, bottom, left }}
      width="180" height="180" viewBox="0 0 180 180"
      fill="none"
    >
      {/* Hexagon outlines */}
      {[[90,90,55],[90,90,38],[130,60,28],[50,130,28],[140,130,22],[40,60,22]].map(([cx,cy,r],i) => (
        <polygon
          key={i}
          points={Array.from({length:6},(_,k)=>{
            const a = Math.PI/180*(60*k-30);
            return `${cx+r*Math.cos(a)},${cy+r*Math.sin(a)}`;
          }).join(' ')}
          stroke="rgba(0,229,255,0.18)"
          strokeWidth="1"
          fill="none"
        />
      ))}
      {/* Corner accent lines */}
      <line x1="0" y1="20" x2="60" y2="20" stroke="rgba(0,229,255,0.5)" strokeWidth="1.5"/>
      <line x1="0" y1="20" x2="0" y2="80" stroke="rgba(0,229,255,0.5)" strokeWidth="1.5"/>
      <line x1="8" y1="26" x2="8" y2="74" stroke="rgba(0,229,255,0.2)" strokeWidth="1"/>
    </svg>
  );
}

// ─── ECG DOM overlay ───────────────────────────────────────────────────────────
function ECGOverlay() {
  const canvasRef = useRef();
  const offset = useRef(0);
  const rafRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 260; canvas.height = 60;

    const draw = () => {
      ctx.clearRect(0, 0, 260, 60);
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 1.5;
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#00e5ff';
      ctx.beginPath();

      const mid = 30;
      for (let x = 0; x < 260; x++) {
        const norm = ((x / 260) + offset.current) % 1;
        let y = mid;
        if      (norm > 0.28 && norm < 0.30) y = mid - 8;
        else if (norm > 0.30 && norm < 0.34) y = mid + 22;
        else if (norm > 0.34 && norm < 0.38) y = mid - 16;
        else if (norm > 0.38 && norm < 0.42) y = mid + 5;
        else if (norm > 0.55 && norm < 0.58) y = mid - 4;
        else if (norm > 0.58 && norm < 0.60) y = mid + 6;
        else y = mid + Math.sin(norm * Math.PI * 6) * 1.5;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      offset.current += 0.004;
      rafRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <div className="bm-ecg-wrap">
      <canvas ref={canvasRef} className="bm-ecg-canvas" />
    </div>
  );
}

// ─── Info Popup ────────────────────────────────────────────────────────────────
function InfoPopup({ muscleId, onClose }) {
  if (!muscleId) return null;
  const m = MUSCLES[muscleId];
  const tips = MUSCLE_TIPS[muscleId] || [];
  return (
    <div className="bm-popup" role="dialog" aria-label={`${m.name} info`}>
      <button className="bm-popup__close" onClick={onClose}>✕</button>
      <div className="bm-popup__badge" style={{ color: m.color, borderColor: m.color + '55', background: m.color + '15' }}>
        {m.group}
      </div>
      <div className="bm-popup__name">{m.name}</div>
      <div className="bm-popup__latin">{m.latin}</div>
      <div className="bm-popup__divider" style={{ background: `linear-gradient(to right, ${m.color}44, transparent)` }} />
      <div className="bm-popup__section">Training Tips</div>
      <ul className="bm-popup__tips">
        {tips.map((t, i) => (
          <li key={i} className="bm-popup__tip">
            <span className="bm-popup__tip-num" style={{ color: m.color }}>0{i+1}</span>
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ─── Muscle Sidebar ────────────────────────────────────────────────────────────
const GROUPS = ['Chest','Shoulder','Arms','Back','Core','Legs'];

function Sidebar({ selectedMuscle, onSelect }) {
  return (
    <div className="bm-sidebar">
      {GROUPS.map(group => {
        const groupMuscles = Object.entries(MUSCLES).filter(([, m]) => m.group === group);
        if (!groupMuscles.length) return null;
        return (
          <div key={group} className="bm-sidebar__group">
            <div className="bm-sidebar__group-label">{group}</div>
            {groupMuscles.map(([id, m]) => (
              <button
                key={id}
                className={`bm-sidebar__btn ${selectedMuscle === id ? 'bm-sidebar__btn--active' : ''}`}
                style={{ '--m-color': m.color }}
                onClick={() => onSelect(selectedMuscle === id ? null : id)}
              >
                <span className="bm-sidebar__dot" style={{ background: m.color }} />
                {m.name}
              </button>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// ─── Scene ─────────────────────────────────────────────────────────────────────
function Scene({ selectedMuscle, hoveredMuscle, onMuscleClick, onMuscleHover, autoRotate }) {
  return (
    <>
      {/* Lighting */}
      <ambientLight color={0x0a1833} intensity={3} />
      <directionalLight color={0x4080ff} intensity={4} position={[2, 4, 3]} />
      <directionalLight color={0x00e5ff} intensity={1.5} position={[-3, 1, -2]} />
      <pointLight color={0x0050ff} intensity={2} position={[0, -2, -2]} />
      <pointLight color={0x00e5ff} intensity={1} position={[0, 2, 2]} />

      {/* Body */}
      <AnatomicalBody
        selectedMuscle={selectedMuscle}
        hoveredMuscle={hoveredMuscle}
        onMuscleClick={onMuscleClick}
        onMuscleHover={onMuscleHover}
      />

      {/* Controls */}
      <OrbitControls
        enablePan={false}
        minDistance={1.2}
        maxDistance={5}
        dampingFactor={0.06}
        enableDamping
        autoRotate={autoRotate}
        autoRotateSpeed={0.7}
        target={[0, 0, 0]}
      />

      {/* Post processing */}
      <EffectComposer>
        <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} intensity={1.2} />
        <Vignette eskil={false} offset={0.2} darkness={0.7} />
      </EffectComposer>
    </>
  );
}

// ─── Main Export ────────────────────────────────────────────────────────────────
export default function BodyMap3D({ className = '' }) {
  const [selectedMuscle, setSelectedMuscle] = useState(null);
  const [hoveredMuscle, setHoveredMuscle] = useState(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [view, setView] = useState('front');

  const handleMuscleClick = (id) => {
    setSelectedMuscle(prev => prev === id ? null : id);
    setAutoRotate(false);
  };

  const handleSidebarSelect = (id) => {
    setSelectedMuscle(id);
    if (id) setAutoRotate(false);
  };

  return (
    <div className={`bm-root ${className}`}>
      {/* Background grid */}
      <div className="bm-bg-grid" />

      {/* Hex corner decorations */}
      <HexCorner top="0" left="0" />
      <HexCorner top="0" right="0" style={{ transform: 'scaleX(-1)' }} />

      {/* ECG line */}
      <ECGOverlay />

      {/* Header */}
      <div className="bm-header">
        <div className="bm-header__title">
          <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden>
            <ellipse cx="11" cy="5" rx="3" ry="3.5" stroke="#00e5ff" strokeWidth="1.2" fill="rgba(0,229,255,0.1)"/>
            <path d="M7 10c0-2.5 1.8-4 4-4s4 1.5 4 4v1l1.5 8H5.5L7 11Z" stroke="#00e5ff" strokeWidth="1.2" fill="rgba(0,229,255,0.08)"/>
            <circle cx="11" cy="10" r="1.2" fill="#00e5ff" opacity="0.8"/>
          </svg>
          JL Body Atlas
        </div>
        <div className="bm-header__sub">Interactive Muscle Reference · Click any muscle</div>
      </div>

      {/* Canvas */}
      <div className="bm-canvas-wrap">
        <Canvas
          camera={{ position: [0, 0.1, 2.4], fov: 42 }}
          gl={{ antialias: true, alpha: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.3 }}
          style={{ background: 'transparent' }}
          onPointerMissed={() => { setSelectedMuscle(null); }}
        >
          <Scene
            selectedMuscle={selectedMuscle}
            hoveredMuscle={hoveredMuscle}
            onMuscleClick={handleMuscleClick}
            onMuscleHover={setHoveredMuscle}
            autoRotate={autoRotate}
          />
        </Canvas>
      </div>

      {/* Sidebar */}
      <Sidebar selectedMuscle={selectedMuscle} onSelect={handleSidebarSelect} />

      {/* Info popup */}
      <InfoPopup muscleId={selectedMuscle} onClose={() => setSelectedMuscle(null)} />

      {/* Controls bar */}
      <div className="bm-controls">
        <button className={`bm-ctrl ${view === 'front' ? 'bm-ctrl--active' : ''}`}
          onClick={() => setView('front')}>Front</button>
        <button className={`bm-ctrl ${view === 'back' ? 'bm-ctrl--active' : ''}`}
          onClick={() => setView('back')}>Back</button>
        <div className="bm-ctrl-divider" />
        <button className={`bm-ctrl ${autoRotate ? 'bm-ctrl--active' : ''}`}
          onClick={() => setAutoRotate(p => !p)}>Auto Rotate</button>
        <button className="bm-ctrl" onClick={() => { setSelectedMuscle(null); setAutoRotate(true); }}>
          Reset
        </button>
      </div>

      {/* Hover label */}
      {hoveredMuscle && !selectedMuscle && (
        <div className="bm-hover-label" style={{ color: MUSCLES[hoveredMuscle]?.color }}>
          {MUSCLES[hoveredMuscle]?.name}
        </div>
      )}
    </div>
  );
}
