# Run MyThings AI Recommender with Docker Compose

The Compose configuration starts two services: the **Python FastAPI/E5** API and **Ollama/Qwen2.5 7B**. No independent Ollama installation is required when using Compose. The React demo is run separately with Node.js/npm.

## Requirements

Docker with Compose, internet access to download images/models, enough RAM/disk for Qwen2.5 7B and E5. The provided Compose file **does not configure GPU acceleration**; CPU-only Qwen inference can be very slow and may exceed the current per-pair 30-second HTTP timeout.

## Start the containers

Run from the repository root:

```powershell
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b
docker compose ps
```

Ollama model weights are stored in a named volume. The API's E5 model is fetched on first usage and cached in a separate named volume.

Open `http://127.0.0.1:8000/docs` for interactive API documentation. `GET /health` reports API liveness, **not** that Ollama has downloaded Qwen or is ready.

To use the React demo, in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Networking

Inside a container `127.0.0.1` refers to that container, not another service. `compose.yaml` sets `OLLAMA_URL=http://ollama:11434/api/chat` for the API; Docker DNS resolves `ollama` to the model service. Running outside Docker defaults to `http://127.0.0.1:11434/api/chat`.

Only the FastAPI endpoint is mapped to the host, on loopback port 8000. The Ollama port is not exposed publicly.

## Limitations and troubleshooting

- `depends_on` controls startup order, not model availability. Run the pull command before sending requests.
- Current Qwen requests are made **per cart-item/candidate pair**. Model unavailability or slow inference may return `uncertain` (and these fallback results can be cached).
- The first E5 prediction may download embeddings; network access is needed for first use.
- Inspect logs with `docker compose logs -f api` or `docker compose logs -f ollama`.
- Stop containers with `docker compose down`. Avoid `down -v` unless you intend to erase downloaded model volumes.
- This is a developer demonstration, not production deployment: no TLS, authentication, rate limiting, readiness gates, observability, or queueing.
- Container startup and model compatibility must be tested on the target machine; simply supplying a Dockerfile does not verify they work in that environment.
