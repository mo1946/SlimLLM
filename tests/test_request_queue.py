"""Controlled CPU fixtures for queue races, cancellation and shutdown; no GPU results."""

import asyncio
import math
import threading
import unittest

from request_queue import InferenceQueue, QueueOverloaded, QueueStopped, QueueWaitTimeout


class BlockingEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.order = []
        self.active = 0
        self.peak_active = 0
        self.threads = set()
        self.last_metrics = {}

    def extract(self, text):
        self.order.append(text)
        self.threads.add(threading.get_ident())
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if text == 'first':
                self.started.set()
                if not self.release.wait(3):
                    raise RuntimeError('Unit fixture release timed out')
            if text == 'error':
                raise MemoryError('fixture')
            self.last_metrics = {'generated_tokens': len(self.order)}
            return text
        finally:
            self.active -= 1


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = BlockingEngine()
        self.queue = InferenceQueue(self.engine, max_waiting=2, wait_timeout=1)
        self.queue.start()
        self.tasks = []

    async def asyncTearDown(self):
        self.engine.release.set()
        await self.queue.close()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def submit(self, text):
        task = asyncio.create_task(self.queue.submit(text))
        self.tasks.append(task)
        return task

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.005)

    async def first(self):
        task = self.submit('first')
        await self.until(self.engine.started.is_set)
        return task

    async def test_fifo_capacity_and_single_thread(self):
        first = await self.first()
        second = self.submit('second')
        await self.until(lambda: len(self.queue.pending) == 1)
        third = self.submit('third')
        await self.until(lambda: len(self.queue.pending) == 2)
        with self.assertRaises(QueueOverloaded):
            await self.queue.submit('overflow')
        self.engine.release.set()
        results = await asyncio.gather(first, second, third)
        self.assertEqual([r[0] for r in results], ['first', 'second', 'third'])
        self.assertEqual([r[1]['generated_tokens'] for r in results], [1, 2, 3])
        self.assertEqual(self.engine.order, ['first', 'second', 'third'])
        self.assertEqual(self.engine.peak_active, 1)
        self.assertEqual(len(self.engine.threads), 1)

    async def test_wait_timeout_frees_slot_and_skips_expired_job(self):
        first = await self.first()
        self.queue.wait_timeout = .03
        with self.assertRaises(QueueWaitTimeout):
            await self.queue.submit('expired')
        self.assertEqual(len(self.queue.pending), 0)
        self.assertEqual(self.engine.order, ['first'])
        third = self.submit('third')
        await self.until(lambda: len(self.queue.pending) == 1)
        self.engine.release.set()
        await asyncio.gather(first, third)
        self.assertEqual(self.engine.order, ['first', 'third'])
        self.assertEqual(self.queue.timed_out, 1)

    async def test_cancel_waiting_or_running_does_not_overlap_model_calls(self):
        first = await self.first()
        waiting = self.submit('cancelled')
        await self.until(lambda: len(self.queue.pending) == 1)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(len(self.queue.pending), 0)
        third = self.submit('third')
        await self.until(lambda: len(self.queue.pending) == 1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.queue.active)
        self.assertEqual(self.engine.order, ['first'])
        self.engine.release.set()
        await third
        self.assertEqual(self.engine.order, ['first', 'third'])
        self.assertEqual(self.engine.peak_active, 1)

    async def test_shutdown_rejects_waiters_and_waits_for_active_thread(self):
        first = await self.first()
        waiting = self.submit('pending')
        await self.until(lambda: len(self.queue.pending) == 1)
        closing = asyncio.create_task(self.queue.close())
        with self.assertRaises(QueueStopped):
            await waiting
        self.assertFalse(closing.done())
        with self.assertRaises(QueueStopped):
            await self.queue.submit('new')
        self.engine.release.set()
        await asyncio.gather(first, closing)
        self.assertEqual(self.engine.order, ['first'])

    async def test_failure_does_not_kill_consumer(self):
        with self.assertRaises(MemoryError):
            await self.queue.submit('error')
        result, _, _ = await self.queue.submit('ok')
        self.assertEqual(result, 'ok')

    def test_invalid_queue_limits_fail(self):
        for count, seconds in ((0, 1), (33, 1), (8, 0), (8, math.inf), (8, math.nan)):
            with self.assertRaises(ValueError):
                InferenceQueue(self.engine, count, seconds)


if __name__ == '__main__':
    unittest.main()
