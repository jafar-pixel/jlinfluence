# BodyMap3D

Interactive 3D medical body map React component — matches the rotating blue holographic human reference.

## Visual style

- Deep navy `#020d1a` background with dot grid
- Cyan hex-grid corner UI decorations
- Animated ECG/heartbeat line overlay
- Blue translucent anatomical body (`MeshPhysicalMaterial` transmission)
- Cyan rim lighting + bloom post-processing glow
- 13 clickable muscle regions with per-muscle color accents
- Glassmorphism info popup with training tips
- Collapsible sidebar muscle list

## Install dependencies

```bash
npm install three @react-three/fiber @react-three/drei @react-three/postprocessing
```

## Usage

```jsx
import { BodyMap3D } from './components/body-map';

// Full-page
<div style={{ width: '100%', height: '100vh' }}>
  <BodyMap3D />
</div>

// Dashboard card
<div style={{ width: 900, height: 640 }}>
  <BodyMap3D />
</div>
```

## Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `className` | `string` | `''` | Extra class on root element |

## Muscles included

| ID | Name | Group |
|---|---|---|
| `pectorals` | Pectorals | Chest |
| `deltoids` | Deltoids | Shoulder |
| `biceps` | Biceps | Arms |
| `triceps` | Triceps | Arms |
| `trapezius` | Trapezius | Back |
| `latissimus` | Latissimus Dorsi | Back |
| `rectus` | Rectus Abdominis | Core |
| `obliques` | Obliques | Core |
| `quadriceps` | Quadriceps | Legs |
| `hamstrings` | Hamstrings | Legs |
| `glutes` | Glutes | Legs |
| `calves` | Calves | Legs |
| `gastrocnemius` | Gastrocnemius | Legs |

## Extending

Add muscles in the `MUSCLES` object at the top of `BodyMap3D.jsx`:

```js
myMuscle: {
  name: 'Display Name',
  latin: 'Musculus latinus',
  group: 'Chest',           // must match a group in GROUPS array
  color: '#ff6b6b',
  pos: [x, y, z],           // Three.js world position
  size: 0.1,                // sphere radius
}
```

Add training tips in `MUSCLE_TIPS`:
```js
myMuscle: ['Tip one', 'Tip two', 'Tip three'],
```
