/**
 * JLSystemFlowMap.jsx
 * Converted from jl_system_flow_map.html — JL System Phase 14 Flow Map
 * Assessment pipeline: Speed Testing · Timing Gate · Vertical Jump · F-V Profiling · SSC Analysis
 */

import React, { useState } from 'react';
import './JLSystemFlowMap.css';

// ─── Data ──────────────────────────────────────────────────────────────────

const FLOW_MODULES = [
  {
    id: 'speed_testing',
    label: 'Speed Testing',
    icon: '⚡',
    color: '#3b82f6',
    inputs: ['gate_crossed (elapsed_ms)', 'sprint_split_ready'],
    outputs: ['split_times_ms', 'top_speed_kmh', 'acceleration_profile'],
    description: 'Captures sprint splits via timing gates across 10/20/30/40m distances. Feeds directly into F-V profiling.',
  },
  {
    id: 'timing_gate',
    label: 'Timing Gate',
    icon: '⏱',
    color: '#6366f1',
    inputs: ['gate_crossed (elapsed_ms)'],
    outputs: ['sprint_split_ready', 'fv_input_ready'],
    description: 'Hardware interface layer — translates gate break events into timestamped split packets for the assessment pipeline.',
  },
  {
    id: 'vertical_jump',
    label: 'Vertical Jump',
    icon: '📈',
    color: '#10b981',
    inputs: ['takeoff_timestamp_ms', 'landing_timestamp_ms'],
    outputs: ['jump_height_cm', 'rsi_score', 'flight_time_ms', 'cmj_summary'],
    description: 'Computes jump height from flight time. Derives RSI (Reactive Strength Index) for SSC quality scoring.',
  },
  {
    id: 'fv_profiling',
    label: 'F-V Profiling',
    icon: '📊',
    color: '#f59e0b',
    inputs: ['sprint_split_ready', 'body_mass_kg', 'height_m'],
    outputs: ['fv_slope', 'pmax_watts', 'f0_newtons', 'v0_ms', 'fv_profile'],
    description: 'Derives the Force-Velocity profile from sprint split data and athlete anthropometrics. Identifies power deficits.',
  },
  {
    id: 'ssc_analysis',
    label: 'SSC Analysis',
    icon: '🔄',
    color: '#ef4444',
    inputs: ['cmj_summary', 'rsi_score', 'fv_profile'],
    outputs: ['ssc_classification', 'training_rx', 'asymmetry_flag'],
    description: 'Stretch-Shortening Cycle classification. Combines jump, sprint, and F-V data to generate training prescriptions.',
  },
];

const FLOW_CONNECTIONS = [
  { from: 'timing_gate', to: 'speed_testing', label: 'gate_crossed' },
  { from: 'speed_testing', to: 'fv_profiling', label: 'split_times' },
  { from: 'vertical_jump', to: 'ssc_analysis', label: 'cmj_summary' },
  { from: 'fv_profiling', to: 'ssc_analysis', label: 'fv_profile' },
];

const SIGNAL_FIELDS = [
  { key: 'takeoff_timestamp_ms', source: 'Jump Mat / IMU', type: 'timestamp', flow: 'vertical_jump' },
  { key: 'landing_timestamp_ms', source: 'Jump Mat / IMU', type: 'timestamp', flow: 'vertical_jump' },
  { key: 'cmj_summary', source: 'Vertical Jump module', type: 'object', flow: 'ssc_analysis' },
  { key: 'gate_crossed (elapsed_ms)', source: 'Timing Gate hardware', type: 'number', flow: 'speed_testing' },
  { key: 'sprint_split_ready', source: 'Timing Gate module', type: 'event', flow: 'fv_profiling' },
];

// ─── Sub-components ────────────────────────────────────────────────────────

function ModuleCard({ module, isActive, onClick }) {
  return (
    <button
      className={`jl-flow-card ${isActive ? 'jl-flow-card--active' : ''}`}
      style={{ '--card-color': module.color }}
      onClick={() => onClick(module.id)}
      aria-pressed={isActive}
    >
      <span className="jl-flow-card__icon">{module.icon}</span>
      <span className="jl-flow-card__label">{module.label}</span>
      <span className="jl-flow-card__arrow">›</span>
    </button>
  );
}

function ModuleDetail({ module, onClose }) {
  return (
    <div className="jl-flow-detail" style={{ '--card-color': module.color }}>
      <button className="jl-flow-detail__close" onClick={onClose} aria-label="Close">✕</button>
      <div className="jl-flow-detail__header">
        <span className="jl-flow-detail__icon">{module.icon}</span>
        <div>
          <div className="jl-flow-detail__title">{module.label}</div>
          <div className="jl-flow-detail__badge" style={{ color: module.color }}>Phase 14 Module</div>
        </div>
      </div>
      <p className="jl-flow-detail__desc">{module.description}</p>
      <div className="jl-flow-detail__cols">
        <div>
          <div className="jl-flow-detail__section-title">Inputs</div>
          <ul className="jl-flow-signal-list">
            {module.inputs.map(i => (
              <li key={i} className="jl-flow-signal-list__item jl-flow-signal-list__item--in">
                <span className="jl-flow-signal-list__dot" />
                <code>{i}</code>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="jl-flow-detail__section-title">Outputs</div>
          <ul className="jl-flow-signal-list">
            {module.outputs.map(o => (
              <li key={o} className="jl-flow-signal-list__item jl-flow-signal-list__item--out">
                <span className="jl-flow-signal-list__dot" />
                <code>{o}</code>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function SignalTable() {
  return (
    <div className="jl-flow-table-wrap">
      <div className="jl-flow-section-title">Signal Fields</div>
      <table className="jl-flow-table">
        <thead>
          <tr>
            <th>Key</th>
            <th>Source</th>
            <th>Type</th>
            <th>Consumed By</th>
          </tr>
        </thead>
        <tbody>
          {SIGNAL_FIELDS.map(f => (
            <tr key={f.key}>
              <td><code>{f.key}</code></td>
              <td>{f.source}</td>
              <td><span className={`jl-flow-type jl-flow-type--${f.type}`}>{f.type}</span></td>
              <td>{FLOW_MODULES.find(m => m.id === f.flow)?.label ?? f.flow}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────

export default function JLSystemFlowMap({ className = '' }) {
  const [activeId, setActiveId] = useState(null);

  const activeModule = FLOW_MODULES.find(m => m.id === activeId);

  const handleCardClick = (id) => {
    setActiveId(prev => (prev === id ? null : id));
  };

  return (
    <div className={`jl-flow-root ${className}`}>
      {/* Header */}
      <div className="jl-flow-header">
        <div className="jl-flow-header__title">
          <span className="jl-flow-header__badge">Phase 14</span>
          JL System — Flow Map
        </div>
        <div className="jl-flow-header__sub">
          Assessment pipeline · Click a module to inspect signals
        </div>
      </div>

      {/* Pipeline */}
      <div className="jl-flow-pipeline">
        {FLOW_MODULES.map((mod, i) => (
          <React.Fragment key={mod.id}>
            <ModuleCard
              module={mod}
              isActive={activeId === mod.id}
              onClick={handleCardClick}
            />
            {i < FLOW_MODULES.length - 1 && (
              <div className="jl-flow-connector">
                <div className="jl-flow-connector__line" />
                <div className="jl-flow-connector__arrow">▶</div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Detail panel */}
      {activeModule && (
        <ModuleDetail
          module={activeModule}
          onClose={() => setActiveId(null)}
        />
      )}

      {/* Connection legend */}
      <div className="jl-flow-connections">
        <div className="jl-flow-section-title">Data Connections</div>
        <div className="jl-flow-connections__grid">
          {FLOW_CONNECTIONS.map(c => {
            const fromMod = FLOW_MODULES.find(m => m.id === c.from);
            const toMod = FLOW_MODULES.find(m => m.id === c.to);
            return (
              <div key={`${c.from}-${c.to}`} className="jl-flow-conn-item">
                <span className="jl-flow-conn-item__from" style={{ color: fromMod?.color }}>
                  {fromMod?.label}
                </span>
                <span className="jl-flow-conn-item__arrow">→</span>
                <span className="jl-flow-conn-item__to" style={{ color: toMod?.color }}>
                  {toMod?.label}
                </span>
                <code className="jl-flow-conn-item__key">{c.label}</code>
              </div>
            );
          })}
        </div>
      </div>

      {/* Signal table */}
      <SignalTable />
    </div>
  );
}
