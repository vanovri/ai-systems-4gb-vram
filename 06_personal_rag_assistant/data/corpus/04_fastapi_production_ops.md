# FastAPI Production Operations for AI Services

FastAPI pairs Pydantic request validation with ASGI concurrency to serve retrieval and machine learning endpoints reliably.

## Health Endpoints and Readiness Probes

A production FastAPI `/health` endpoint should expose machine-readable readiness fields including `status`, `index_loaded`, `num_chunks`, and the active embedding `backend`. Container orchestrators and load balancers poll `/health` before routing live traffic to a worker instance.

## Event Loop Safety and Threadpool Offloading

FastAPI runs `async def` route handlers directly on the asyncio event loop, whereas standard `def` route handlers are automatically executed in Starlette's AnyIO worker threadpool. Offloading synchronous CPU-bound FAISS queries or PDF parsing to a threadpool prevents blocking the asyncio event loop and preserves low tail latency for concurrent API callers.

## Secret Hygiene

API keys such as `OPENAI_API_KEY` must always be injected via environment variables or secret managers and never hardcoded in source files or persisted inside serialized vector index manifests.
