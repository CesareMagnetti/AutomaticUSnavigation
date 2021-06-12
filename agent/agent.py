import numpy as np
import random
from collections import namedtuple, deque
from .Qnetworks import SimpleQNetwork as QNetwork
import torch
import torch.nn as nn
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 8          # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, Nagents, seed, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (list/tuple): dimension of each state (CxWxH)
            action_size (int): dimension of each action
            Nagents (int): how many agents we have
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.Nagents = Nagents
        self.seed = random.seed(seed)

        # get the devce for gpu dependencies
        self.device = torch.device('cuda' if kwargs.pop('use_cuda', False) else 'cpu')

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, Nagents, seed, **kwargs).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, Nagents, seed, **kwargs).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # loss
        self.loss = kwargs.pop('loss', nn.SmoothL1Loss())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, actions, reward, next_state):
        # Save experience in replay memory
        self.memory.add(state, actions, reward, next_state)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = state.float().unsqueeze(0).to(self.device)
        print("state: ", state.shape)
        self.qnetwork_local.eval()
        with torch.no_grad():
            Qs = self.qnetwork_local(state)
            print("Qs: ", Qs.shape)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return [self.mapActionToIncrement(Q.max()[1].item()) for Q in Qs]
        else:
            return [self.mapActionToIncrement(random.choice(np.arange(self.action_size)))for _ in range(self.Nagents)] 

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # compute and minimize the loss for each agent
        Qs = self.qnetwork_local(states)
        MaxQs = self.qnetwork_target(next_states)

        for Q, A, MaxQ in zip(Qs, actions.T, MaxQs):
            # gather Q value for the action taken
            Q = Q.gather(1, A)
            # get the value of the best action in the next state
            # detach to only optimize local network
            MaxQ = MaxQ.max(1)[0].detach()
            # backup the expected value of this action  
            Qhat = rewards + gamma*MaxQ
            # evalauate TD error
            loss = self.loss(Q, Qhat)
            # optimize local network parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    @staticmethod
    def mapActionToIncrement(action):
        """ 
        using the following convention:
        0: +1 in x coordinate
        1: -1 in x coordinate
        2: +1 in y coordinate
        3: -1 in y coordinate
        4: +1 in z coordinate
        5: -1 in z coordinate
        """

        if action == 0:
            incr = np.array([1, 0, 0])
        elif action == 1:
            incr = np.array([-1, 0, 0])
        elif action == 2:
            incr = np.array([0, 1, 0])
        elif action == 3:
            incr = np.array([0, -1, 0])
        elif action == 4:
            incr = np.array([0, 0, 1])
        elif action == 5:
            incr = np.array([0, 0, -1])
        else:
            raise ValueError('unknown action: %d, legal actions: [0, 1, 2, 3, 4, 5]'%action)
        
        return incr


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # note that action is going to contain the action of each agent
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.cat([e.state for e in experiences if e is not None]).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.cat([e.next_state for e in experiences if e is not None]).float().to(self.device)
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)