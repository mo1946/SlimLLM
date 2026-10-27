"""Opt-in local HTTP client for one WSL vLLM engine; never loads GPU model weights."""

import hashlib
import json
import os
import threading
import time
from urllib.parse import urlparse

import httpx

from inference import Extractor, InputTooLong, InvalidModelOutput, validate_output
from runtime import ROOT, configure
from schema import AfterSalesTicket


SUPPORTED_VLLM_VERSION = '0.11.2'
REQUEST_PROTOCOL_REVISION = 'structured-outputs-v1'
STRUCTURED_OUTPUT_BACKEND = 'guidance'


class BackendUnavailable(Exception):
    pass


class BackendTimeout(Exception):
    pass


class BackendRequestRejected(Exception):
    def __init__(self, status_code: int):
        super().__init__('Upstream rejected the request configuration.')
        self.status_code = status_code


class VllmExtractor(Extractor):
    def __init__(self, structured_outputs: str | None = None):
        mode = (os.getenv('SLIMLLM_VLLM_STRUCTURED_OUTPUTS', 'json_schema')
                if structured_outputs is None else structured_outputs)
        if mode not in ('json_schema', 'off'):
            raise ValueError('SLIMLLM_VLLM_STRUCTURED_OUTPUTS must be json_schema or off.')
        super().__init__()
        self._base_fingerprint = self.fingerprint
        schema = AfterSalesTicket.model_json_schema()
        exported = json.loads((ROOT / 'data/ticket.schema.json').read_text(encoding='utf-8'))
        if schema != exported:
            raise ValueError('Pydantic schema differs from data/ticket.schema.json.')
        self._structured_output_mode = mode
        # Keep an immutable snapshot; request copies retain the original property order.
        self._structured_schema_json = json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
        canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        self._schema_sha256 = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        self.url = os.getenv('SLIMLLM_VLLM_URL', 'http://127.0.0.1:8001').rstrip('/')
        parsed = urlparse(self.url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.path or parsed.username:
            raise ValueError('vLLM must use a local HTTP endpoint without credentials or a path.')
        self.model_name = 'slimllm-lora'
        self.client = None
        self.version = None
        self.counter_lock = threading.Lock()
        self.tokenizer_lock = threading.Lock()
        self.fingerprint = self._cache_fingerprint()

    @property
    def structured_output_mode(self) -> str:
        return self._structured_output_mode

    @property
    def structured_schema_sha256(self) -> str | None:
        return self._schema_sha256 if self.structured_output_mode == 'json_schema' else None

    def _cache_fingerprint(self) -> str:
        identity = {'base': self._base_fingerprint, 'backend': 'vllm',
                    'version': self.version, 'adapter_alias': self.model_name,
                    'request_protocol': REQUEST_PROTOCOL_REVISION,
                    'structured_output_mode': self.structured_output_mode,
                    'structured_schema_sha256': self._schema_sha256,
                    'structured_output_backend': STRUCTURED_OUTPUT_BACKEND}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()

    def load(self):
        configure()
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False)
        self.client = httpx.Client(base_url=self.url, timeout=httpx.Timeout(55, connect=3),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8), trust_env=False)
        try:
            models = self.client.get('/v1/models')
            models.raise_for_status()
            if self.model_name not in {row['id'] for row in models.json()['data']}:
                raise ValueError('Configured vLLM server is missing the required LoRA model alias.')
            version = self.client.get('/version')
            version.raise_for_status()
            self.version = version.json()['version']
            if self.version != SUPPORTED_VLLM_VERSION:
                raise ValueError('Unvalidated vLLM version; expected 0.11.2. No automatic protocol fallback.')
            self.fingerprint = self._cache_fingerprint()
            if not self.check_health():
                raise BackendUnavailable('vLLM is not healthy.')
        except Exception:
            self.close()
            raise

    def check_health(self):
        try:
            self.ready = self.client is not None and self.client.get('/health', timeout=2).status_code == 200
        except httpx.RequestError:
            self.ready = False
        return self.ready

    def build_request(self, text: str) -> dict:
        """Build an isolated payload without sending HTTP or incrementing calls."""
        if self.tokenizer is None:
            raise BackendUnavailable()
        with self.tokenizer_lock:
            rendered = self.tokenizer.apply_chat_template(
                [{'role':'system','content':self.prompt}, {'role':'user','content':text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokens = list(self.tokenizer.encode(rendered, add_special_tokens=False))
        if len(tokens) > self.max_input_tokens:
            raise InputTooLong()
        payload = {'model': self.model_name, 'prompt': tokens, 'max_tokens': self.max_new_tokens,
                   'temperature': 0, 'seed': 42, 'n': 1, 'stream': False}
        if self.structured_output_mode == 'json_schema':
            payload['structured_outputs'] = {'json': json.loads(self._structured_schema_json)}
        return payload

    def extract_with_metrics(self, text):
        if not self.ready:
            raise BackendUnavailable()
        request = self.build_request(text)
        with self.counter_lock:
            self.calls += 1
        start = time.perf_counter()
        try:
            response = self.client.post('/v1/completions', json=request)
        except httpx.TimeoutException:
            raise BackendTimeout() from None
        except httpx.RequestError:
            self.ready = False
            raise BackendUnavailable() from None
        if response.status_code in (429, 503):
            raise BackendUnavailable()
        if response.status_code in (400, 422):
            raise BackendRequestRejected(response.status_code)
        if response.status_code != 200:
            raise InvalidModelOutput('Upstream rejected the request.')
        try:
            payload = response.json()
            choice = payload['choices'][0]
            count = payload['usage']['completion_tokens']
            if type(count) is not int or count < 1 or count >= self.max_new_tokens or choice['finish_reason'] != 'stop':
                raise ValueError('Invalid or truncated completion.')
            result = validate_output(choice['text'], text)
        except (ValueError, KeyError, IndexError, TypeError):
            raise InvalidModelOutput('Invalid vLLM completion.') from None
        # HTTP upstream time includes its queue/batching; do not label it CUDA generation time.
        return result, {'generated_tokens':count, 'backend_request_seconds':time.perf_counter()-start}

    def extract(self, text):
        return self.extract_with_metrics(text)[0]

    def close(self):
        self.ready = False
        if self.client is not None:
            self.client.close()
            self.client = None
        self.tokenizer = None
