# JL Pulse — BioDigital Content API helpers

Two tools for pulling your BioDigital library (My Human + collections) into JL Pulse.

## 1. CLI — one-off library dump

```bash
export BIODIGITAL_CLIENT_ID=your_client_id
export BIODIGITAL_CLIENT_SECRET=your_client_secret

python fetch_library.py                          # all content → library.json
python fetch_library.py --search "step up"       # filter by title/id
python fetch_library.py --filter module          # only full 3D models
python fetch_library.py --gender male
```

Produces `library.json`:
```json
{
  "fetched_at": "2026-04-18T21:00:00Z",
  "count": 24,
  "items": [
    {
      "id": "production/maleAdult/box_step_up_02",
      "title": "Box Step Up",
      "type": "module",
      "gender": "male",
      "thumbnail": "https://human.biodigital.com/thumbs/modules/…jpg",
      "embed_url": "https://human.biodigital.com/viewer/?id=…&dk=…",
      "is_animated": true
    }
  ]
}
```

## 2. Server — proxy for the live JL Pulse UI

Keeps the `client_secret` on the server; browsers only hit `/api/library`.

### Run locally
```bash
pip install -r requirements.txt
export BIODIGITAL_CLIENT_ID=...
export BIODIGITAL_CLIENT_SECRET=...
uvicorn server:app --port 8080 --reload
```

### Deploy on Vercel (same project as jl-pulse)
1. `vercel --prod` in this folder (or add it as a serverless `api/` route on your existing Vercel project).
2. In Vercel → Project → Settings → Environment Variables, add:
   - `BIODIGITAL_CLIENT_ID`
   - `BIODIGITAL_CLIENT_SECRET`
   - `BIODIGITAL_DEV_KEY` (defaults to your existing key)

### Endpoints
| Route                                    | Returns                                                |
| ---------------------------------------- | ------------------------------------------------------ |
| `GET /api/health`                        | `{ ok, has_credentials }`                              |
| `GET /api/library`                       | All My Library content, flattened                      |
| `GET /api/library?search=squat&type=module&gender=male&animated=true` | filtered |
| `GET /api/collections`                   | Your saved collections                                 |
| `GET /api/collections/{id}/content`      | Content in a collection                                |

### Front-end usage
```js
const res  = await fetch('/api/library?type=module&search=squat');
const data = await res.json();
data.items.forEach(scene => {
  // scene.id          → "production/maleAdult/box_step_up_02"
  // scene.embed_url   → ready-to-set iframe src
  // scene.thumbnail   → preview JPEG
});
```

## Getting credentials

1. Sign in at [developer.biodigital.com](https://developer.biodigital.com).
2. Open your app (the one issuing `dk=ab649fdf…`).
3. Add the **Content API** scope if it isn't already enabled.
4. Copy **Client ID** and **Client Secret** from the API Credentials panel.
5. **Never commit the secret.** Use env vars only.

## Where models like `box_step_up_02` come from

- **Public/production content** (e.g. `production/maleAdult/box_step_up_02`) — browse [human.biodigital.com/content](https://human.biodigital.com/content), open a scene, copy the `id=` from its URL.
- **Your own saved content** — save a scene via the BioDigital Human web app → appears at `myhuman/...` → discoverable via `/api/library`.
- **Team content** — linking your account to a team surfaces team scenes in `/myhuman` too.
