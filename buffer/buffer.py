import numpy as np
from collections import deque
import pickle

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
        self.buffer_size = buffer_size  
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, transition):
        """Add a new experience to memory."""
        self.memory.append(transition)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # 2. sample experiences
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=True)
        experiences = [self.memory[i] for i in indices] # random lookup in a deque is more efficient
        # experiences = np.random.choice(self.memory, size=self.batch_size, replace=False)
        # reorganize batch
        batch = zip(*experiences)
        return batch
    
    def save(self, fname):
        pickle.dump(self.memory, open('{}_buffer.pkl'.format(fname), 'wb'))

    def load(self, fname):
        self.memory = pickle.load(open('{}_buffer.pkl'.format(fname), 'rb'))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# ==== PRIORITIZED EXPERIENCE ====
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(self, buffer_size, batch_size)
        self.prob_alpha = prob_alpha
        self.priorities = deque(maxlen=buffer_size)
    
    def add(self, transition):
        # 1. add to experience
        super(PrioritizedReplayBuffer, self).add(transition)
        # 2. add to buffer with max probability to incentivize exploration of new transitions
        max_prio = max(self.priorities) if len(self.priorities)>0 else 1.0
        self.priorities.append(max_prio)
    
    def sample(self, beta=0.4):
        # 1. get prioritized distribution
        probs = np.array(self.priorities)
        probs = probs ** self.prob_alpha
        probs /= probs.sum()
        # 2. sample experiences
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=True, p=probs)
        experiences = [self.memory[i] for i in indices] # random lookup in a deque is more efficient
        # 3. reorganize batch
        batch = zip(*experiences)
        # 3. sample weights for bias correction of the gradient
        N = len(self.memory)
        max_weight = (probs.min() * N) ** (-beta)
        weights = ((N*probs[indices])**(-beta))/max_weight
        return batch, weights, indices
    
    def update_priorities(self, batch_indices, batch_priorities, eps=10e-5):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + eps
    
    def save(self, fname):
        pickle.dump(self.memory, open('{}_buffer.pkl'.format(fname), 'wb'))
        pickle.dump(self.priorities, open('{}_priorities.pkl'.format(fname), 'wb'))
    
    def load(self, fname):
        self.memory = pickle.load(open('{}_buffer.pkl'.format(fname), 'rb'))
        self.priorities = pickle.load(open('{}_priorities.pkl'.format(fname), 'rb'))


class RecurrentPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """A prioritized replay buffer that samples a history of states and next states recusrvively for partially observable environments.
    We limit the recurvive property to states and next states without caring about actions, rewards and dones. We do this because we will
    pass a sequence of states to the qnetwork, process it with some recurrent/transformer architecture and we only want to predict the current action.
    More sophisticated versions would add recursive property to actions and rewards"""
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6, history_length=10):
        super(RecurrentPrioritizedReplayBuffer, self).__init__(self, buffer_size, batch_size, prob_alpha)
        self.history_length = history_length-1 #we want the full input to be of length history length, so we subtract 1
        # lookback will guide you to a particular entry in self.memory to recover useful information (state and next_state in our case)
        self.lookback = deque(maxlen=buffer_size)
        # we use the flag -1 to indicate a first time-step (episode changed) we do not want to lookback overlapping episodes
        self.lookup_index = -1 
    
    def add(self, transition, is_first_time_step):
        super(RecurrentPrioritizedReplayBuffer, self).add(transition)
        # add lookback index (signal first time-step with a -1 so we don't cross over episodes)
        if is_first_time_step:
            self.lookup_index = -1
        self.lookback.append(self.lookup_index)
        self.lookup_index+=1
    
    def sample(self, beta=0.4):
        # sample normally
        batch, weights, indices = super(RecurrentPrioritizedReplayBuffer, self).asampledd(beta)
        # lookup to previous time step for the history of states and next_states for each batch index
        for i, index in enumerate(indices):
            # initialize state next_state history with the sampled state and next state
            states, next_states = [batch[0][i]], [batch[3][i]]
            # perform n lookback steps to get previous states and next states
            lookback_index = index
            for t in range(self.history_length):
                # change the lookback index if we are not at the start of a new episode
                if lookback_index != -1:
                    lookback_index = self.lookback[lookback_index]
                # add new state and next_states, if we are at the start it will pad with the same state and next_state (sequences will be padded to the left this way)
                states = self.memory[lookback_index][0] + states
                next_states = self.memory[lookback_index][0] + next_states
            # replace the original single state and next_state entry in the batch with these sequential ones
            batch[0][i] = states # B x L x 3 x 3
            batch[3][i] = next_states # B x L x 3 x 3
        return batch, weights, indices