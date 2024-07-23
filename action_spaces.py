import numpy as np

simple_actions = {
    0: np.array([.1,0,0], dtype=np.float64), # front
    1: np.array([0,.1,0], dtype=np.float64), # right
    2: np.array([-.1,0,0], dtype=np.float64), # back
    3: np.array([0,-.1,0], dtype=np.float64), # left
    4: np.array([0,0,-.1], dtype=np.float64) # down
}