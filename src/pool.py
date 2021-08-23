import numpy as np


class CustomPool:
    """Class for storing and providing samples of different stages of growth."""
    def __init__(self, seed, size):
        self.size = size
        self.slots = np.repeat([seed], size, 0)
        self.seed = seed

    def commit(self, batch):
        """Replace existing slots with a batch."""
        indices = batch["indices"]
        for i, x in enumerate(batch["x"]):
            if (x[:, :, 3] > 0.1).any():  # Avoid committing dead image
                self.slots[indices[i]] = x.copy()

    def sample(self, c):
        """Retrieve a batch from the pool."""
        indices = np.random.choice(self.size, c, False)
        batch = {
            "indices": indices,
            "x": self.slots[indices]
        }
        return batch
