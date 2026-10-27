"""Offline Transformers/LoRA inference and strict output validation."""

import gc
import hashlib
import json
import os
import time

from pydantic import ValidationError

from baseline import MODEL_ID, REVISION, adapter_provenance, load_fp16_model
from evaluate import strict_json
from runtime import ROOT, configure, file_hash, project_path
from schema import AfterSalesTicket


class InputTooLong(ValueError):
    pass


class InvalidModelOutput(ValueError):
    pass


def validate_output(raw: str, source: str) -> AfterSalesTicket:
    try:
        ticket = AfterSalesTicket.model_validate(strict_json(raw))
    except (ValueError, TypeError, ValidationError):
        raise InvalidModelOutput('Model output is not a valid ticket.') from None
    for name in ('order_id', 'product_name', 'reason'):
        value = getattr(ticket, name)
        if value is not None and value not in source:
            raise InvalidModelOutput('Model output contains an ungrounded text field.')
    return ticket


class Extractor:
    def __init__(self):
        self.ready = False
        self.calls = 0
        self.last_metrics: dict = {}
        self.model = None
        self.tokenizer = None
        self.max_input_tokens = 2048
        self.max_new_tokens = 384
        self.model_path = project_path('.cache/huggingface/qwen3-1.7b')
        self.adapter = adapter_provenance('outputs/qlora/final_adapter')
        self.prompt = json.loads((ROOT / 'reports/baseline/run.json').read_text(encoding='utf-8'))['prompt']
        identity = {'model': MODEL_ID, 'revision': REVISION, 'adapter': self.adapter,
            'schema': file_hash(ROOT / 'data/ticket.schema.json'), 'prompt': self.prompt,
            'max_input_tokens': self.max_input_tokens, 'max_new_tokens': self.max_new_tokens,
            'do_sample': False, 'enable_thinking': False, 'dtype': 'float16'}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def load(self) -> None:
        configure()
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_DEACTIVATE_ASYNC_LOAD='1', USE_HUB_KERNELS='0')
        self.model, self.tokenizer = load_fp16_model(self.model_path, self.adapter['path'])
        self.ready = True

    def extract(self, text: str) -> AfterSalesTicket:
        if not self.ready:
            raise RuntimeError('Model not ready.')
        import torch

        messages = [{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': text}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        if inputs.input_ids.shape[-1] > self.max_input_tokens:
            raise InputTooLong('Request exceeds the model input token limit.')
        try:
            self.last_metrics = {}
            torch.cuda.reset_peak_memory_stats()
            inputs = inputs.to('cuda')
            self.calls += 1
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated = output[0, inputs.input_ids.shape[-1]:]
            self.last_metrics = {'generated_tokens': len(generated), 'generation_seconds': elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            if len(generated) >= self.max_new_tokens:
                raise InvalidModelOutput('Model response reached its token limit.')
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            return validate_output(raw, text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise MemoryError('GPU memory unavailable.') from None

    def close(self) -> None:
        self.ready = False
        self.model = None
        self.tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
