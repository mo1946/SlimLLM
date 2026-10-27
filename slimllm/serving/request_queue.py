"""Bounded FIFO; local GPU models use one consumer, remote HTTP can use a few."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math


class QueueOverloaded(Exception):
    pass


class QueueWaitTimeout(Exception):
    pass


class QueueStopped(Exception):
    pass


@dataclass(eq=False)
class Job:
    text: str
    enqueued: float
    started: asyncio.Future
    result: asyncio.Future


class InferenceQueue:
    def __init__(self, engine, max_waiting: int = 8, wait_timeout: float = 30, workers: int = 1):
        if not 1 <= max_waiting <= 32 or not math.isfinite(wait_timeout) or not 0 < wait_timeout <= 120:
            raise ValueError('Queue max_waiting must be 1..32 and wait_timeout must be >0..120 seconds.')
        if workers not in (1, 2, 3, 4):
            raise ValueError('Queue workers must be 1..4; Transformers always uses 1.')
        self.engine = engine
        self.max_waiting = max_waiting
        self.wait_timeout = wait_timeout
        self.pending: deque[Job] = deque()
        self.wakeup = asyncio.Event()
        self.worker_count = workers
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='slimllm-model')
        self.running = 0
        self.closed = False
        self.worker = None
        self.workers = []
        self.rejected = 0
        self.timed_out = 0
        self.peak_waiting = 0

    def start(self):
        self.workers = [asyncio.create_task(self._consume(), name=f'slimllm-fifo-{i}')
                        for i in range(self.worker_count)]
        self.worker = self.workers[0]

    @property
    def active(self):
        return self.running > 0

    async def submit(self, text: str):
        if self.closed:
            raise QueueStopped()
        if len(self.pending) >= self.max_waiting:
            self.rejected += 1
            raise QueueOverloaded()
        loop = asyncio.get_running_loop()
        job = Job(text, loop.time(), loop.create_future(), loop.create_future())
        self.pending.append(job)
        self.peak_waiting = max(self.peak_waiting, len(self.pending))
        self.wakeup.set()
        try:
            try:
                await asyncio.wait_for(asyncio.shield(job.started), self.wait_timeout)
            except asyncio.TimeoutError:
                # Deadline and consumer may be scheduled in the same loop iteration.
                if not job.started.done():
                    self._remove(job)
                    self.timed_out += 1
                    raise QueueWaitTimeout() from None
            return await asyncio.shield(job.result)
        except asyncio.CancelledError:
            self._remove(job)
            raise

    def _remove(self, job):
        if job in self.pending:
            self.pending.remove(job)
        job.started.cancel()
        job.result.cancel()

    def _infer(self, text):
        if hasattr(self.engine, 'extract_with_metrics'):
            return self.engine.extract_with_metrics(text)
        result = self.engine.extract(text)
        return result, dict(getattr(self.engine, 'last_metrics', {}))

    async def _consume(self):
        loop = asyncio.get_running_loop()
        while True:
            await self.wakeup.wait()
            while self.pending:
                job = self.pending.popleft()
                if job.result.done():
                    continue
                waited = loop.time() - job.enqueued
                job.started.set_result(None)
                if waited >= self.wait_timeout:
                    self.timed_out += 1
                    job.result.set_exception(QueueWaitTimeout())
                    continue
                self.running += 1
                try:
                    # Each consumer awaits its call; never cancel running CUDA work.
                    result, metrics = await loop.run_in_executor(self.executor, self._infer, job.text)
                    if not job.result.done():
                        job.result.set_result((result, metrics, waited))
                except Exception as exc:
                    if not job.result.done():
                        job.result.set_exception(exc)
                finally:
                    self.running -= 1
            self.wakeup.clear()
            if self.closed:
                return

    async def close(self):
        self.closed = True
        while self.pending:
            job = self.pending.popleft()
            if not job.result.done():
                job.started.set_result(None)
                job.result.set_exception(QueueStopped())
        self.wakeup.set()
        if self.workers:
            await asyncio.gather(*self.workers)
        self.executor.shutdown(wait=True, cancel_futures=True)

    def status(self):
        return {'waiting': len(self.pending), 'max_waiting': self.max_waiting,
                'wait_timeout_seconds': self.wait_timeout, 'active': self.active,
                'running': self.running, 'workers': self.worker_count,
                'peak_waiting': self.peak_waiting, 'rejected_full': self.rejected,
                'wait_timeouts': self.timed_out}
