# MyThings AI Recommender — frontend demo

A React + TypeScript + Vite demonstration of a shopping cart and AI-generated complementary product suggestions.

This is **synthetic demo data**, not the MyThings production app or catalog. Products, merchant ID, prices, and displayed labels are defined in `src/App.tsx`.

## Run

Start the Python API and Ollama as described in the [root README](../README.md).

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to `http://127.0.0.1:8000`.

The demo lets you add/remove products, switch Arabic/English, and request AI recommendations automatically when the cart changes. The API returns product IDs and heuristic scores; the frontend resolves IDs against the synthetic catalog to show names and sample prices.

## Verify

```powershell
npm run build
npm run lint
```

If Ollama or FastAPI is unavailable, recommendation requests will fail, but the local product catalog can still render.

## Production integration boundary

Catalog management, availability, pricing, and authentication would belong to the production commerce backend. This frontend is a demonstration only.
