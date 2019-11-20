from game import Vector2
from models.FeedForwardDNN import FeedForwardDNN
import numpy as np

gen = np.random.default_rng()

agent_args = {
    'view_size': Vector2(15, 15),
    'layer_config': [
        ('conv', 3, (3, 3)),
        ('linear', 64),  ('act', 'sigmoid'),
        ('linear', 64),  ('act', 'sigmoid'),
    ],
}

model = FeedForwardDNN(**agent_args, generator=gen)