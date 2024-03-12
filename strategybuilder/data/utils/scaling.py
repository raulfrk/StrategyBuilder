from asyncio import Protocol
from typing import Any


class Scaler(Protocol):
    def fit_transform(self, x, y=None, **fit_params) -> Any: ...

    def fit(self, x, y=None, **fit_params) -> "Scaler": ...

    def inverse_transform(self, x) -> Any: ...