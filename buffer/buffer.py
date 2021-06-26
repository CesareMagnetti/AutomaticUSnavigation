import numpy as np
from collections import deque

# ========== REPLAY BUFFER CLASS =========
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    BUFFER_SIZE = None
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): capacity of the replay buffer
            batch_size (int): batch size sampled each time we call self.sample()

        """
        BUFFER_SIZE = buffer_size  
        self.batch_size = batch_size

    # instanciating memory here will make sure that memory is shared among instances
    memory = deque(maxlen=BUFFER_SIZE)

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = (state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = np.random.choice(self.memory, size=self.batch_size, replace=False)
        # reorganize batch
        batch = zip(*experiences)
        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# ==== PRIORITIZED EXPERIENCE ====
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6):
        ReplayBuffer.__init__(self, buffer_size, batch_size)
        self.prob_alpha = prob_alpha
        self.capacity = capacity
    
    # instanciating priorities here will make sure that they are shared among instances
    priorities = deque(maxlen=BUFFER_SIZE)
    
    def add(self, state, action, reward, next_state):
        # 1. add to experience
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state)
        # 2. add to buffer with max probability to incentivize exploration of new transitions
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities.append(max_prio)
    
    def sample(self, beta=0.4):
        # 1. get prioritized distribution
        probs = np.array(self.priorities)
        probs = self.priorities ** self.prob_alpha
        probs /= probs.sum()
        # 2. sample experiences
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
        experiences = [self.memory[i] for i in indices]
        # 3. reorganize batch
        batch = zip(*experiences)
        # 3. sample weights for bias correction
        N = len(self.memory)
        max_weight = (probs.min() * N) ** (-beta)
        weights = ((N*probs[indices])**(-beta))/max_weight
        return batch, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio