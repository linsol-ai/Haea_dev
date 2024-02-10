import abc
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

class ParameterScheduler(abc.ABC):
    """Base class for parameter schedulers."""

    @abc.abstractmethod
    def step(self) -> None:
        """Update the parameters of the scheduler."""

    @abc.abstractmethod
    def get_value(self) -> float:
        """Get the current value of the parameter."""


class LinearScheduler(ParameterScheduler):
    """A scheduler that linearly interpolates between two values."""

    def __init__(
        self, start: float, end: float, steps: int
    ) -> None:
        """Init the linear scheduler.

        Args:
            start: The start value of the parameter.
            end: The end value of the parameter.
            steps: The number of steps to take.
            warmup: Fraction of steps to warm up. The scheduler will start
                interpolating from `start` to `end` at `int(warmup * steps)` step.
            cooldown: Fraction of steps to cool down. The scheduler will be
                interpolating from `end` to `start` till `1 - int(cooldown * steps)` step.
        """
        super().__init__()
        if steps < 1:
            raise ValueError("`steps` must be at least 1.")
        
        self._value = start
        self._start_step = 0
        self._end_step = steps
        self._step_size = (end - start) / (self._end_step - self._start_step)
        self._step = 0
        self.decay_sc = decay_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    def step(self, global_steps) -> None:  # noqa: D102
        if self._start_step <= global_steps < self._end_step:
            self._value += self._step_size
        else:


    def get_value(self) -> float:  # noqa: D102
        return self._value


class ExponentialScheduler(ParameterScheduler):
    def __init__(
        self, start: float, min: float, anneal_rate: float = 1e-6
    ) -> None:
        """Init the linear scheduler.

        Args:
            start: The start value of the parameter.
            end: The end value of the parameter.
            steps: The number of steps to take.
            warmup: Fraction of steps to warm up. The scheduler will start
                interpolating from `start` to `end` at `int(warmup * steps)` step.
            cooldown: Fraction of steps to cool down. The scheduler will be
                interpolating from `end` to `start` till `1 - int(cooldown * steps)` step.
        """
        super().__init__()
        self._value = start
        self._min = min
        self._anneal_rate = anneal_rate

    def step(self, global_steps) -> None:  # noqa: D102
        self._value = max(self._value * math.exp(-self._anneal_rate * global_steps), self._min)

    def get_value(self) -> float:  # noqa: D102
        return self._value
