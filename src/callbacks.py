import logging
from abc import ABC

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):

    def on_train_begin(self, **ctx) -> None:
        pass

    def on_train_end(self, **ctx) -> None:
        pass

    def on_epoch_begin(self, epoch: int, **ctx) -> None:
        pass

    def on_epoch_end(self, epoch: int, **ctx) -> None:
        pass

    def on_batch_end(self, global_step: int, **ctx) -> None:
        pass


class CallbackRunner:

    def __init__(self, callbacks: list[TrainingCallback] | None = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, **ctx):
        for cb in self.callbacks:
            cb.on_train_begin(**ctx)

    def on_train_end(self, **ctx):
        for cb in self.callbacks:
            cb.on_train_end(**ctx)

    def on_epoch_begin(self, epoch: int, **ctx):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch=epoch, **ctx)

    def on_epoch_end(self, epoch: int, **ctx):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch=epoch, **ctx)

    def on_batch_end(self, global_step: int, **ctx):
        for cb in self.callbacks:
            cb.on_batch_end(global_step=global_step, **ctx)
