import random
from collections import deque

# ========== REPLAY BUFFER CLASS =========
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): capacity of the replay buffer
            batch_size (int): batch size sampled each time we call self.sample()

        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # (named tuple does note cope well with multiprocessing, storing a standard tuple instead)
        #self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states"])
    
    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = (state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # reorganize batch
        batch = zip(*experiences)
        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)