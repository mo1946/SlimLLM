"""Single-worker FastAPI extraction service; local models only."""

import argparse
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from inference import Extractor, InputTooLong, InvalidModelOutput, validate_output
from request_queue import InferenceQueue, QueueOverloaded, QueueStopped, QueueWaitTimeout
from response_cache import RedisExactCache
from schema import AfterSalesTicket
from vllm_backend import VllmExtractor, BackendRequestRejected, BackendTimeout, BackendUnavailable

logger = logging.getLogger('slimllm.api')


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    text: str = Field(min_length=1, max_length=2000, pattern=r'\S',
                      description='Single after-sales ticket, up to 2000 characters.')


def create_app(engine_factory=None, cache_factory=None, *, max_waiting=None, wait_timeout=None, backend=None) -> FastAPI:
    selected_backend = backend or os.getenv('SLIMLLM_BACKEND', 'transformers')
    if selected_backend not in ('transformers', 'vllm'):
        raise ValueError('SLIMLLM_BACKEND must be transformers or vllm.')
    selected_factory = engine_factory or (VllmExtractor if selected_backend == 'vllm' else Extractor)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = selected_factory()
        app.state.engine = engine
        queue = InferenceQueue(engine,
            max_waiting=int(os.getenv('SLIMLLM_QUEUE_MAX_WAITING', '8')) if max_waiting is None else max_waiting,
            wait_timeout=float(os.getenv('SLIMLLM_QUEUE_WAIT_SECONDS', '30')) if wait_timeout is None else wait_timeout,
            workers=int(os.getenv('SLIMLLM_VLLM_MAX_INFLIGHT', '2')) if selected_backend == 'vllm' else 1)
        app.state.inference_queue = queue
        cache = None
        try:
            await run_in_threadpool(engine.load)
            cache = (cache_factory(engine.fingerprint) if cache_factory else
                     RedisExactCache(engine.fingerprint, os.getenv('SLIMLLM_REDIS_URL')))
            app.state.cache = cache
            await run_in_threadpool(cache.ping)
            queue.start()
            yield
        finally:
            try:
                await queue.close()
            finally:
                try:
                    if cache is not None:
                        await run_in_threadpool(cache.close)
                finally:
                    await run_in_threadpool(engine.close)

    app = FastAPI(title='SlimLLM', version='0.4.0', lifespan=lifespan)

    @app.get('/health', description='Reports configured decoding policy, not proof of grammar enforcement or semantic accuracy.')
    async def health():
        engine = app.state.engine
        cache = app.state.cache
        if selected_backend == 'vllm':
            await run_in_threadpool(engine.check_health)
        mode = getattr(engine, 'structured_output_mode', 'off') if selected_backend == 'vllm' else 'none'
        result = {'status': 'ready' if engine.ready else 'unavailable',
            'backend': selected_backend, 'model': 'Qwen/Qwen3-1.7B',
            'structured_output_mode': mode,
            'structured_schema_sha256': getattr(engine, 'structured_schema_sha256', None)
                if selected_backend == 'vllm' and mode == 'json_schema' else None,
            'adapter_loaded': engine.ready, 'busy': app.state.inference_queue.active,
            'queue': app.state.inference_queue.status(),
            'model_fingerprint': engine.fingerprint,
            'inference_calls': engine.calls,
            'cache': 'disabled' if not cache.enabled else 'available' if cache.available else 'unavailable'}
        return JSONResponse(result, status_code=200 if engine.ready else 503)

    @app.post('/extract', response_model=AfterSalesTicket, responses={
        502: {'description': 'Invalid or truncated model output, or upstream request configuration rejected.'},
        503: {'description': 'Queue full, service stopping, or model/backend/GPU unavailable.'},
        504: {'description': 'Queue wait or backend request timed out.'},
    })
    async def extract(request: ExtractRequest, response: Response):
        engine = app.state.engine
        cache = app.state.cache
        if not engine.ready:
            raise HTTPException(503, detail={'code': 'model_not_ready'})
        raw = await run_in_threadpool(cache.get, request.text)
        if raw is not None:
            try:
                result = validate_output(raw, request.text)
                response.headers['X-SlimLLM-Cache'] = 'HIT'
                response.headers['X-SlimLLM-Queue-Wait-Seconds'] = '0'
                return result
            except InvalidModelOutput:
                pass
        try:
            result, metrics, waited = await app.state.inference_queue.submit(request.text)
            for name, value in metrics.items():
                response.headers['X-SlimLLM-' + name.replace('_', '-')] = str(value)
            response.headers['X-SlimLLM-Queue-Wait-Seconds'] = str(waited)
        except QueueOverloaded:
            raise HTTPException(503, detail={'code': 'queue_full'}, headers={'Retry-After': '1'}) from None
        except QueueWaitTimeout:
            raise HTTPException(504, detail={'code': 'queue_wait_timeout'}) from None
        except QueueStopped:
            raise HTTPException(503, detail={'code': 'service_stopping'}) from None
        except BackendTimeout:
            raise HTTPException(504, detail={'code': 'backend_timeout'}) from None
        except BackendRequestRejected as exc:
            logger.warning('Upstream request rejected (status=%d, type=BackendRequestRejected)', exc.status_code)
            raise HTTPException(502, detail={'code': 'backend_request_rejected'}) from None
        except BackendUnavailable:
            raise HTTPException(503, detail={'code': 'backend_unavailable'}) from None
        except InputTooLong:
            raise HTTPException(422, detail={'code': 'input_token_limit'}) from None
        except InvalidModelOutput:
            raise HTTPException(502, detail={'code': 'invalid_model_output'}) from None
        except MemoryError:
            raise HTTPException(503, detail={'code': 'gpu_memory_unavailable'}) from None
        except Exception as exc:
            logger.error('Inference failed (%s)', type(exc).__name__)
            raise HTTPException(500, detail={'code': 'inference_failed'}) from None
        await run_in_threadpool(cache.set, request.text, result.model_dump_json())
        response.headers['X-SlimLLM-Cache'] = 'MISS' if cache.enabled else 'DISABLED'
        return result

    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1, access_log=False)
