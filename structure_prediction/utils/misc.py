import random

import numpy as np
import paddle


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


