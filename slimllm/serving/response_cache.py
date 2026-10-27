"""Optional, bounded-time Redis exact cache. No in-process second cache."""

import hashlib


class RedisExactCache:
    def __init__(
        self, namespace: str, url: str | None = None, client=None, ttl: int = 3600
    ):
        self.namespace = namespace
        self.ttl = ttl
        self.enabled = bool(url) or client is not None
        self.available = False
        self.client = client
        if url and client is None:
            import redis

            self.client = redis.Redis.from_url(
                url,
                decode_responses=True,
                protocol=2,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
                retry_on_timeout=False,
            )

    def key(self, text: str) -> str:
        return f'slimllm:v1:{self.namespace}:{hashlib.sha256(text.encode("utf-8")).hexdigest()}'

    def ping(self) -> bool:
        if not self.enabled or self.client is None:
            return False
        try:
            self.available = bool(self.client.ping())
        except Exception:
            self.available = False
        return self.available

    def get(self, text: str) -> str | None:
        if not self.enabled or self.client is None:
            return None
        try:
            value = self.client.get(self.key(text))
            self.available = True
            return (
                value
                if isinstance(value, str) and len(value.encode("utf-8")) <= 8192
                else None
            )
        except Exception:
            self.available = False
            return None

    def set(self, text: str, value: str) -> None:
        if not self.enabled or self.client is None or len(value.encode("utf-8")) > 8192:
            return
        try:
            self.client.setex(self.key(text), self.ttl, value)
            self.available = True
        except Exception:
            self.available = False

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
