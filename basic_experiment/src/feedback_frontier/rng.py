from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


NAMESPACES = {
    "instance", "base", "proposal", "rollout", "scheduler_tie", "bootstrap",
    "width_calibration_bank",
}


@dataclass(frozen=True)
class SeedBook:
    master_seed: int

    def rng(self, namespace: str, *keys: object) -> np.random.Generator:
        if namespace not in NAMESPACES:
            raise ValueError(f"unregistered RNG namespace: {namespace}")
        payload = json.dumps(
            [self.master_seed, namespace, *keys],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        entropy = np.frombuffer(digest, dtype=np.uint32).tolist()
        return np.random.default_rng(np.random.SeedSequence(entropy))
