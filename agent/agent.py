import numpy as np
import random
from collections import namedtuple, deque
from .Qnetworks import SimpleQNetwork as QNetwork
import torch
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, parser, **kwargs):
        """Initialize an Agent object.
        s
        Params that we will use:
        ======
            state_size (list/tuple): dimension of each state (CxWxH)
            parser.action_size (int): dimension of each action
            parser.n_agents (int): how many agents we have
            parser.seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = parser.action_size
        self.n_agents = parser.n_agents
        self.seed = random.seed(parser.seed)

        # get the devce for gpu dependencies
        self.device = torch.device('cuda' if kwargs.pop('use_cuda', False) else 'cpu')

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, self.action_size, self.agents, self.seed, **kwargs).to(self.device)
        print("Q Network instanciated: (%d parameters)"%self.qnetwork_local.count_parameters())
        print(self.qnetwork_local)
        self.qnetwork_target = QNetwork(state_size, self.action_size, self.n_agents, self.seed, **kwargs).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # discount factor
        self.gamma = parser.gamma
        # tau for soft target network update
        self.tau = parser.tau
        # lr for q networks
        self.lr = parser.learning_rate
        # learn from buffer every ``update_every`` steps
        self.update_every = parser.update_every
        # loss
        self.loss = parser.loss

        # Replay memory
        self.batch_size = parser.batch_size
        self.buffer_size = parser.buffer_size
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, actions, reward, next_state):
        # convert increment actions back to their ids
        actions = [self.mapIncrementToAction(a) for a in actions]
        # Save experience in replay memory
        self.memory.add(state, actions, reward, next_state)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences, self.gamma)
                return loss
        else:
            return None

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = state.float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            Qs = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return [self.mapActionToIncrement(torch.argmax(Q, dim=1).item()) for Q in Qs]
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

        self.optimizer.zero_grad()
        for Q, A, MaxQ in zip(Qs, actions.T, MaxQs):
            # gather Q value for the action taken
            Q = Q.gather(1, A.unsqueeze(-1))
            # get the value of the best action in the next state
            # detach to only optimize local network
            MaxQ = MaxQ.max(1)[0].detach().unsqueeze(-1)
            # backup the expected value of this action  
            Qhat = rewards.unsqueeze(-1) + gamma*MaxQ
            # evalauate TD error
            loss = self.loss(Q, Qhat)
            # retain graph because we will backprop multiple times through the backbone cnn
            loss.backward(retain_graph=True)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)   

        return loss.item()                  

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

    @staticmethod
    def mapIncrementToAction(incr):
        """ 
        using the following convention:
        0: +1 in x coordinate
        1: -1 in x coordinate
        2: +1 in y coordinate
        3: -1 in y coordinate
        4: +1 in z coordinate
        5: -1 in z coordinate
        """

        if incr[0] == 1:
            action = 0
        elif incr[0] == -1:
            action = 1
        elif incr[1] == 1:
            action = 2
        elif incr[1] == -1:
            action = 3
        elif incr[2] == 1:
            action = 4
        elif incr[2] == -1:
            action = 5
        else:
            raise ValueError('unknown increment: {}.'.format(incr))
        
        return action


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): cpu or gpu
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.device = device
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

        states = torch.cat([e.state for e in experiences if e is not None]).float().unsqueeze(1).to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.cat([e.next_state for e in experiences if e is not None]).float().unsqueeze(1).to(self.device)
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)