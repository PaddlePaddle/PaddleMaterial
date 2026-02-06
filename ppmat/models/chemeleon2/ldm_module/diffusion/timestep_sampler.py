from abc import ABC, abstractmethod
import numpy as np
import paddle


def create_named_schedule_sampler(name, diffusion):
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size, device):
        w = self.weights()
        assert w is not None, "weights() should not return None"
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = paddle.to_tensor(indices_np, dtype='int64')
        weights_np = 1 / (len(p) * p[indices_np])
        weights = paddle.to_tensor(weights_np, dtype='float32')
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_local_losses(self, local_ts, local_losses):
        batch_sizes = [len(local_ts)]
        max_bs = max(batch_sizes)

        timestep_batches = [local_ts]
        loss_batches = [local_losses]

        timesteps = paddle.concat(timestep_batches, axis=0)[: sum(batch_sizes)]
        losses = paddle.concat(loss_batches, axis=0)[: sum(batch_sizes)]
        self.update_with_all_losses(timesteps, losses)

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss.item()
            else:
                self._loss_history[t, self._loss_counts[t]] = loss.item()
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
