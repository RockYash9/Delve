# Deploying Delve (free, no credit card)

Delve deploys as **one service** — Render hosts the FastAPI backend, and since `api.py` already serves the frontend itself (`static/index.html`, mounted at `/`), there's nothing separate to deploy. One URL, no CORS setup, nothing to keep in sync across two platforms.

## Deploy

1. Go to [render.com](https://render.com) and sign up (GitHub login is easiest).
2. **New +** → **Blueprint** → connect your `delve` GitHub repo. Render detects `render.yaml` automatically and proposes the service.
3. When prompted for environment variables, fill in:
   - `GEMINI_API_KEY` — your real key
   - `TAVILY_API_KEY` — your real key
   - `ALLOWED_ORIGINS` can stay `*` — with everything on one origin, CORS barely comes into play (it only matters for cross-origin requests, and there aren't any here).
4. Click **Apply** / **Create**. First deploy takes a few minutes — installing PyTorch is the slow part, same as it was locally.
5. Once live, Render gives you a URL like `https://delve-xxxx.onrender.com`. That's it — open it in a browser and the full chat UI is right there, talking to the backend on the same origin.

Test the API directly too, if you want:
```
https://delve-xxxx.onrender.com/health
```
Should return `{"status":"ok","active_sessions":0}`.

## The one honest tradeoff

Render's free tier sleeps the service after 15 minutes of inactivity. The first request after that takes 20-50 seconds to wake back up — and since the frontend is served *by* this same service, that means the whole page (not just API calls) is briefly slow to load right after it's been idle.

This isn't hidden or silently broken: `static/index.html` pings `/health` on load with retry/backoff and shows an honest "waking up — this can take up to a minute" banner during that window, rather than leaving you staring at a blank page with no explanation.

## Updating after this

Any future `git push` to `main` auto-redeploys on Render — no manual redeploy steps for code changes, only for environment variable changes (which you'd update directly in the Render dashboard).

## If you ever want to split frontend/backend later

Serving both from Render is the simplest correct setup for this project's scale, and the recommended default. If you later want a dedicated CDN-backed frontend (e.g. for faster cold-load times independent of backend sleep state), the frontend can be redeployed separately to Vercel/Netlify — that just means pointing `static/index.html`'s `API_BASE` at the Render URL explicitly instead of `window.location.origin`, and setting `ALLOWED_ORIGINS` on Render to that frontend's exact domain. Not necessary for now.
