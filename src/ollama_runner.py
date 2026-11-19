"""
Ollama API İstemcisi.
Donanım TPS telemetrisi ile birlikte inference yapar.
"""

import time
import requests
from typing import Dict, Any, Optional, Union


class OllamaRunner:
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")
        self.chat_url = f"{self.host}/api/chat"
        self.tags_url = f"{self.host}/api/tags"

    def is_healthy(self) -> bool:
        try:
            return requests.get(self.tags_url, timeout=3).status_code == 200
        except Exception:
            return False

    def warmup(self, model: str) -> bool:
        """Modeli VRAM'e yükler, böylece ilk gerçek istek timeout'a uğramaz."""
        try:
            r = requests.post(
                self.chat_url,
                json={"model": model, "messages": [{"role": "user", "content": "merhaba"}], "stream": False},
                timeout=180
            )
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            r = requests.get(self.tags_url, timeout=5)
            if r.status_code == 200:
                return [m.get("name") for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        format_schema: Optional[Union[dict, str]] = None,
        seed: Optional[int] = 42,
        context_length: int = 4096,
        timeout: int = 60,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Ollama chat endpoint'ine istek atar.
        
        İki Deneysel Ortamı Destekler:
        1. Prompt-only: format_schema=None (Ollama format parametresi gönderilmez)
        2. Schema-constrained: format_schema=JSON_SCHEMA (Ollama format parametresine JSON Schema verilir)
        
        num_predict: Maksimum token sınırı (sonsuz döngü / context taşmasını önler).
        seed: Tekrarlanabilirlik (reproducibility) için sabit tohum.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": max_tokens,
                "num_ctx": context_length,
            }
        }

        # Sabit seed ekle (Tekrarlanabilirlik için)
        if seed is not None:
            payload["options"]["seed"] = int(seed)

        # Deneysel Ortam 2: Schema-constrained (format parametresi)
        if format_schema is not None:
            payload["format"] = format_schema

        start = time.perf_counter()
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=timeout)
            wall_time = time.perf_counter() - start

            if resp.status_code != 200:
                return self._error_result(wall_time, f"HTTP {resp.status_code}")

            result = resp.json()
            raw_text = result.get("message", {}).get("content", "")

            eval_count = result.get("eval_count", 0)
            eval_duration_ns = result.get("eval_duration", 0)
            eval_duration_sec = eval_duration_ns / 1e9 if eval_duration_ns else wall_time
            tps = (eval_count / eval_duration_sec) if eval_duration_sec > 0 else 0.0

            return {
                "raw_text": raw_text,
                "latency_sec": round(wall_time, 3),
                "eval_count": eval_count,
                "tps": round(tps, 2),
                "success": True,
                "error": None
            }

        except Exception as e:
            wall_time = time.perf_counter() - start
            return self._error_result(wall_time, str(e))

    def _error_result(self, wall_time, error_msg):
        return {
            "raw_text": "",
            "latency_sec": round(wall_time, 3),
            "eval_count": 0,
            "tps": 0.0,
            "success": False,
            "error": error_msg
        }
