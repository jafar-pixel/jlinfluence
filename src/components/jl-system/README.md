# JL System Components

React components converted from `jl_system_flow_map.html` and `jl_hardware_software_map.html`.

## Components

### `JLSystemFlowMap`
Phase 14 assessment pipeline visualization. Shows the full flow from Timing Gate → Speed Testing → F-V Profiling → Vertical Jump → SSC Analysis with interactive module cards, signal I/O details, data connections, and a signal fields table.

```jsx
import { JLSystemFlowMap } from './components/jl-system';

<JLSystemFlowMap className="my-custom-class" />
```

**Props**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | `string` | `''` | Extra class applied to the root element |

---

### `JLHardwareSoftwareMap`
Full hardware ↔ software architecture map. Tabbed interface covering:
- **Wearable Nodes** — nRF5340 node specs, chip lists, phase labels
- **Data Flow** — Assessment and Training pipeline layers
- **API Contracts** — All first-endpoint routes with method badges
- **UI Panels** — Left / Center / Right panel contents + Stride Feedback + IMU Summary cards
- **Payloads** — Sensor Frame, Pulse Target, Pulse Event schemas
- **Cue Modes** — Phase, Symmetry, Fade-Out, Metronome+Error

```jsx
import { JLHardwareSoftwareMap } from './components/jl-system';

<JLHardwareSoftwareMap className="my-custom-class" />
```

**Props**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | `string` | `''` | Extra class applied to the root element |

---

## Styling

Both components are self-contained with scoped CSS files (`JLSystemFlowMap.css`, `JLHardwareSoftwareMap.css`). They use a dark `#0a0f1a` background and are designed to sit inside any dark-theme dashboard.

To override the background, pass a wrapper class:
```css
.my-panel .jl-flow-root,
.my-panel .jl-hw-root {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.08);
}
```
