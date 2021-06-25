import random
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
        experiences = random.sample(self.memory, k=self.batch_size)
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
        self.capacity   = capacity
        self.pos        = 0
    
    # instanciating priorities here will make sure that they are shared among instances
    priorities = deque(maxlen=BUFFER_SIZE)
    
    def add(self, state, action, reward, next_state):
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        # add to experience
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = zip(*samples)
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)