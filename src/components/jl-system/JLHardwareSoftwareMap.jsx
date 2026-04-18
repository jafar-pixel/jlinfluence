/**
 * JLHardwareSoftwareMap.jsx
 * Converted from jl_hardware_software_map.html
 * JL System — Hardware ↔ Software Architecture Map
 */

import React, { useState } from 'react';
import './JLHardwareSoftwareMap.css';

// ─── Data ──────────────────────────────────────────────────────────────────

const WEARABLE_NODES = [
  {
    id: 'hip',
    label: 'Hip Node (Master)',
    location: 'Hip / Pelvis',
    role: 'Rhythm anchor · Trunk/pelvis timing · Central cue',
    chips: ['nRF5340', 'BNO085', 'DRV2605L', 'LRA Motor'],
    phase: 'V1',
    phaseColor: '#3b82f6',
    category: 'prototype',
  },
  {
    id: 'thigh',
    label: 'Thigh Node',
    location: 'Thigh (R)',
    role: 'Stride phase timing · Leg cycle · Phase cue target',
    chips: ['nRF5340', 'BNO085', 'DRV2605L', 'LRA Motor'],
    phase: 'V1',
    phaseColor: '#3b82f6',
    category: 'prototype',
  },
  {
    id: 'wrist',
    label: 'Wrist / Bicep Nodes',
    location: 'Wrist · Bicep',
    role: 'Arm timing cues · Youth & hearing-impaired · Demo visual',
    chips: ['nRF5340', 'BNO085', 'DRV2605L'],
    phase: 'Train',
    phaseColor: '#10b981',
    category: 'training',
  },
  {
    id: 'ankle',
    label: 'Ankle / Foot Nodes',
    location: 'Shank R/L · Foot R/L',
    role: 'Contact precision · Push-off timing · Full assessment',
    chips: ['nRF5340', 'BNO085'],
    phase: 'P2',
    phaseColor: '#f59e0b',
    category: 'phase2',
  },
];

const CAMERA_TIMING = [
  {
    id: 'ipad',
    label: 'iPad / iPhone Camera',
    role: 'MMPOS input · Video capture · 30fps pose',
    badge: 'Primary',
    badgeColor: '#3b82f6',
    icon: '📱',
  },
  {
    id: 'gates',
    label: 'Timing Gates',
    role: 'Sprint splits · F-V profiling · 10/20/30/40m',
    badge: 'Optional',
    badgeColor: '#6366f1',
    icon: '⚡',
  },
];

const DATA_FLOW_LAYERS = [
  {
    id: 'assessment',
    label: 'ASSESSMENT',
    color: '#3b82f6',
    steps: [
      { icon: '📡', layer: 'Camera + IMU', role: 'Capture' },
      { icon: '🖥', layer: 'Frontend', role: 'Live View' },
      { icon: '🗄', layer: 'Backend', role: 'Store + Score' },
      { icon: '🤖', layer: 'AIQ', role: 'Classify + Rx' },
    ],
  },
  {
    id: 'training',
    label: 'TRAINING',
    color: '#10b981',
    steps: [
      { icon: '🗄', layer: 'Backend', role: 'Create Target' },
      { icon: '🖥', layer: 'Frontend', role: 'Send Cue Plan' },
      { icon: '📡', layer: 'Node', role: 'Fire Pulse' },
      { icon: '🗄', layer: 'Backend', role: 'Compare + Track' },
    ],
  },
];

const API_ENDPOINTS = [
  { method: 'POST', route: '/sessions/:id/sensor-frames', purpose: 'Stream IMU packets from node → backend' },
  { method: 'POST', route: '/sessions/:id/pulse-events', purpose: 'Log haptic pulse fired timestamp' },
  { method: 'POST', route: '/sessions/:id/pulse-targets', purpose: 'Send cue plan: mode, baseline, goal, intensity' },
  { method: 'GET',  route: '/sessions/:id/synced-timeline', purpose: 'Merged video + IMU + pulse stream' },
  { method: 'GET',  route: '/sessions/:id/live-status', purpose: 'Node health, battery, sync status' },
  { method: 'POST', route: '/timing-gate/ingest', purpose: 'Sprint splits → auto F-V profile' },
  { method: 'POST', route: '/jump-mat/ingest', purpose: 'Takeoff/landing ts → jump height + RSI' },
];

const UI_PANELS = [
  {
    id: 'left',
    label: 'LEFT PANEL',
    color: '#6366f1',
    items: ['Athlete profile', 'Drill context', 'Prescription focus', 'Sensor status', 'Battery / sync'],
  },
  {
    id: 'center',
    label: 'CENTER PANEL',
    color: '#3b82f6',
    items: ['🎬 Video + MMPOS overlay', '📊 Floating stride card', '📐 IMU summary card', '💓 Baseline vs goal pulse', '⏮ Playback controls'],
  },
  {
    id: 'right',
    label: 'RIGHT PANEL',
    color: '#10b981',
    items: ['Classification label', 'Cue mode control', 'Coaching notes', 'Next action', 'Review / compare'],
  },
];

const PAYLOADS = [
  {
    id: 'sensor_frame',
    label: 'Sensor Frame',
    color: '#3b82f6',
    fields: [
      { key: 'node_id', value: '"hip_01"' },
      { key: 'location', value: '"pelvis"' },
      { key: 'ts_ms', value: '1710000000' },
      { key: 'accel', value: '{x, y, z}' },
      { key: 'gyro', value: '{x, y, z}' },
      { key: 'quat', value: '{w, x, y, z}' },
      { key: 'battery', value: '92' },
    ],
  },
  {
    id: 'pulse_target',
    label: 'Pulse Target',
    color: '#10b981',
    fields: [
      { key: 'node_id', value: '"thigh_r"' },
      { key: 'mode', value: '"phase"' },
      { key: 'baseline_ms', value: '240' },
      { key: 'goal_ms', value: '220' },
      { key: 'intensity', value: '0.7' },
      { key: 'duration_ms', value: '90' },
    ],
  },
  {
    id: 'pulse_event',
    label: 'Pulse Event',
    color: '#f59e0b',
    fields: [
      { key: 'node_id', value: '"thigh_r"' },
      { key: 'ts_ms', value: '1710000021' },
      { key: 'event', value: '"pulse_fired"' },
      { key: 'mode', value: '"phase"' },
    ],
  },
];

const CUE_MODES = [
  {
    id: 'phase',
    label: 'Phase Mode',
    desc: 'Pulse fires at target movement phase · reinforces timing',
    phase: 'V1 · Primary',
    color: '#3b82f6',
  },
  {
    id: 'symmetry',
    label: 'Symmetry Mode',
    desc: 'Fires on weaker side · reduces left/right gap',
    phase: 'V1',
    color: '#6366f1',
  },
  {
    id: 'fadeout',
    label: 'Fade-Out Mode',
    desc: 'Gradually reduce cue intensity as retention builds',
    phase: 'V1',
    color: '#10b981',
  },
  {
    id: 'metro',
    label: 'Metronome + Error',
    desc: 'Consistent beat anchor · fires on error only',
    phase: 'Phase 2',
    color: '#f59e0b',
  },
];

// ─── Sub-components ────────────────────────────────────────────────────────

const METHOD_COLORS = { GET: '#10b981', POST: '#3b82f6', PUT: '#f59e0b', DELETE: '#ef4444' };

function SectionTitle({ children }) {
  return <div className="jl-hw-section-title">{children}</div>;
}

function NodeCard({ node }) {
  const categoryLabel = { prototype: 'Prototype V1', training: 'Training / Demo', phase2: 'Phase 2' }[node.category];
  return (
    <div className="jl-hw-node-card" style={{ '--node-color': node.phaseColor }}>
      <div className="jl-hw-node-card__top">
        <span className="jl-hw-node-card__label">{node.label}</span>
        <span className="jl-hw-node-card__phase" style={{ color: node.phaseColor }}>{node.phase}</span>
      </div>
      <div className="jl-hw-node-card__location">{node.location}</div>
      <div className="jl-hw-node-card__role">{node.role}</div>
      <div className="jl-hw-node-card__chips">
        {node.chips.map(c => (
          <span key={c} className="jl-hw-chip">{c}</span>
        ))}
      </div>
    </div>
  );
}

function DataFlowRow({ flow }) {
  return (
    <div className="jl-hw-flow-row" style={{ '--flow-color': flow.color }}>
      <div className="jl-hw-flow-row__label" style={{ color: flow.color }}>{flow.label}</div>
      <div className="jl-hw-flow-row__steps">
        {flow.steps.map((s, i) => (
          <React.Fragment key={s.layer + i}>
            <div className="jl-hw-flow-step">
              <span className="jl-hw-flow-step__icon">{s.icon}</span>
              <span className="jl-hw-flow-step__layer">{s.layer}</span>
              <span className="jl-hw-flow-step__role">{s.role}</span>
            </div>
            {i < flow.steps.length - 1 && (
              <div className="jl-hw-flow-arrow" style={{ color: flow.color }}>→</div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function PayloadCard({ payload }) {
  return (
    <div className="jl-hw-payload" style={{ '--payload-color': payload.color }}>
      <div className="jl-hw-payload__title" style={{ color: payload.color }}>{payload.label}</div>
      <div className="jl-hw-payload__body">
        {payload.fields.map(f => (
          <div key={f.key} className="jl-hw-payload__field">
            <span className="jl-hw-payload__key">{f.key}</span>
            <span className="jl-hw-payload__sep">:</span>
            <span className="jl-hw-payload__val">{f.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────

export default function JLHardwareSoftwareMap({ className = '' }) {
  const [activeTab, setActiveTab] = useState('nodes');

  const tabs = [
    { id: 'nodes',    label: 'Wearable Nodes' },
    { id: 'dataflow', label: 'Data Flow' },
    { id: 'api',      label: 'API Contracts' },
    { id: 'ui',       label: 'UI Panels' },
    { id: 'payloads', label: 'Payloads' },
    { id: 'cue',      label: 'Cue Modes' },
  ];

  return (
    <div className={`jl-hw-root ${className}`}>
      {/* Header */}
      <div className="jl-hw-header">
        <div className="jl-hw-header__title">
          <span className="jl-hw-header__badge">JL System</span>
          Hardware ↔ Software Map
        </div>
        <div className="jl-hw-header__sub">
          Architecture reference · Wearable nodes · Data flow · API contracts
        </div>
      </div>

      {/* Tabs */}
      <div className="jl-hw-tabs" role="tablist">
        {tabs.map(t => (
          <button
            key={t.id}
            role="tab"
            aria-selected={activeTab === t.id}
            className={`jl-hw-tab ${activeTab === t.id ? 'jl-hw-tab--active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="jl-hw-content" role="tabpanel">

        {/* Wearable Nodes */}
        {activeTab === 'nodes' && (
          <div>
            <SectionTitle>Wearable Nodes — V1 Prototype</SectionTitle>
            <div className="jl-hw-nodes-grid">
              {WEARABLE_NODES.map(n => <NodeCard key={n.id} node={n} />)}
            </div>

            <SectionTitle>Camera + Timing Hardware</SectionTitle>
            <div className="jl-hw-camera-grid">
              {CAMERA_TIMING.map(c => (
                <div key={c.id} className="jl-hw-camera-card">
                  <span className="jl-hw-camera-card__icon">{c.icon}</span>
                  <div>
                    <div className="jl-hw-camera-card__label">{c.label}</div>
                    <div className="jl-hw-camera-card__role">{c.role}</div>
                  </div>
                  <span className="jl-hw-camera-card__badge" style={{ color: c.badgeColor, borderColor: c.badgeColor }}>
                    {c.badge}
                  </span>
                </div>
              ))}
            </div>

            <SectionTitle>Node Edge Logic (nRF5340)</SectionTitle>
            <div className="jl-hw-edge-logic">
              {['IMU read loop', 'Event detect', 'BLE stream', 'Local pulse fire', 'Packet format'].map(item => (
                <div key={item} className="jl-hw-edge-item">
                  <span className="jl-hw-edge-item__dot" />
                  {item}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Data Flow */}
        {activeTab === 'dataflow' && (
          <div>
            <SectionTitle>Data Flow by Layer</SectionTitle>
            <div className="jl-hw-flows">
              {DATA_FLOW_LAYERS.map(f => <DataFlowRow key={f.id} flow={f} />)}
            </div>
          </div>
        )}

        {/* API Contracts */}
        {activeTab === 'api' && (
          <div>
            <SectionTitle>API Contracts — First Endpoints</SectionTitle>
            <div className="jl-hw-api-table-wrap">
              <table className="jl-hw-api-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Route</th>
                    <th>Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {API_ENDPOINTS.map(e => (
                    <tr key={e.route}>
                      <td>
                        <span className="jl-hw-method" style={{ color: METHOD_COLORS[e.method], borderColor: METHOD_COLORS[e.method] + '44', background: METHOD_COLORS[e.method] + '15' }}>
                          {e.method}
                        </span>
                      </td>
                      <td><code className="jl-hw-route">{e.route}</code></td>
                      <td className="jl-hw-purpose">{e.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* UI Panels */}
        {activeTab === 'ui' && (
          <div>
            <SectionTitle>Frontend UI Panels</SectionTitle>
            <div className="jl-hw-panels-grid">
              {UI_PANELS.map(p => (
                <div key={p.id} className="jl-hw-ui-panel" style={{ '--panel-color': p.color }}>
                  <div className="jl-hw-ui-panel__title" style={{ color: p.color }}>{p.label}</div>
                  <ul className="jl-hw-ui-panel__list">
                    {p.items.map(item => (
                      <li key={item} className="jl-hw-ui-panel__item">{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <SectionTitle>Stride Feedback Panel</SectionTitle>
            <div className="jl-hw-stride-panel">
              {[
                ['💓', 'Baseline pulse'],
                ['🎯', 'Goal pulse'],
                ['⏱', 'Current step timing'],
                ['⚖', 'Symmetry %'],
                ['🔄', 'Cue mode'],
              ].map(([icon, label]) => (
                <div key={label} className="jl-hw-stride-item">
                  <span>{icon}</span>
                  <span>{label}</span>
                </div>
              ))}
            </div>

            <SectionTitle>IMU Summary Card</SectionTitle>
            <div className="jl-hw-stride-panel">
              {[
                ['🦶', 'Step timing'],
                ['⏱', 'Contact timing'],
                ['📈', 'Stride consistency'],
                ['⚖', 'Left/right balance'],
                ['📉', 'Rep drift'],
              ].map(([icon, label]) => (
                <div key={label} className="jl-hw-stride-item">
                  <span>{icon}</span>
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Payloads */}
        {activeTab === 'payloads' && (
          <div>
            <SectionTitle>Minimal Payloads</SectionTitle>
            <div className="jl-hw-payloads-grid">
              {PAYLOADS.map(p => <PayloadCard key={p.id} payload={p} />)}
            </div>
          </div>
        )}

        {/* Cue Modes */}
        {activeTab === 'cue' && (
          <div>
            <SectionTitle>Pulse Cue Modes (V1)</SectionTitle>
            <div className="jl-hw-cue-grid">
              {CUE_MODES.map(m => (
                <div key={m.id} className="jl-hw-cue-card" style={{ '--cue-color': m.color }}>
                  <div className="jl-hw-cue-card__header">
                    <span className="jl-hw-cue-card__label">{m.label}</span>
                    <span className="jl-hw-cue-card__phase" style={{ color: m.color }}>{m.phase}</span>
                  </div>
                  <p className="jl-hw-cue-card__desc">{m.desc}</p>
                  <div className="jl-hw-cue-card__bar" style={{ background: m.color }} />
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
