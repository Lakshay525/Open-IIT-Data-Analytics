# Live demo: run locally and deploy on Vercel

The `web/` folder is a complete, self-contained web app: a static front end (`web/public`) and three Python serverless functions (`web/api`, WSGI apps named `app`)
that run the same model as the pipeline. Nothing outside `web/` is needed at run time.

## Run it locally
```bash
pip install numpy pandas            # the only runtime dependencies
python web/dev_server.py            # http://127.0.0.1:8000
```
The dev server serves `web/public` and routes `/api/*` to the same handlers Vercel runs.

## What the live demo does
| Tab | What happens | Endpoint |
|---|---|---|
| Inspect | Browse all 2,880 addresses on a pan/zoom map: old pin, our pin, calibrated circles, survey truth, directions, reason codes | static JSON |
| Locate an address | Type any address (or click a held-out example). The parser, pin model and calibrated radii run live and return a pin, circle, tier and offline directions in about a second | `POST /api/locate` |
| Teach a visit | Confirm a borrower at a spot (or replay a real visit); the unvisited neighbours on the same street re-estimate, with no retraining | `POST /api/visit` |

## Deploy on Vercel
1. Push the repository to GitHub.
2. On vercel.com choose **Add New -> Project** and import the repository.
3. Settings: **Root Directory = `web`**, **Framework Preset = Other**, leave Build Command empty, **Output Directory = `public`**.
4. Deploy. Open `https://<project>.vercel.app/api/health` - it should answer `{"ok":true,"addresses":2880,"visits":5578}`.
5. Open the site and run the three demo tabs once to warm the functions before presenting (first call after idle takes a few seconds).

Notes
- The functions need about 1 GB of memory and up to 30 s on a cold start (set in `web/vercel.json`).
- If the API is ever unreachable the page still works in static mode (Inspect, charts, results); only the Locate and Teach tabs need the API.
- To refresh the bundle after re-running the pipeline: `python -m geocoder.webassets` (also part of `python run_pipeline.py`), then commit `web/`.
- Dependencies for the functions are pinned in `web/requirements.txt` (numpy 2.x, pandas 3.x, Python 3.11+).
