from __future__ import annotations

from abc import ABCMeta, abstractmethod

import paddle


class HeadInterface(metaclass=ABCMeta):
    @property
    def use_amp(self):
        return False

    @abstractmethod
    def forward(
        self, data: dict[str, paddle.Tensor], emb: dict[str, paddle.Tensor]
    ) -> dict[str, paddle.Tensor]:
        raise NotImplementedError
