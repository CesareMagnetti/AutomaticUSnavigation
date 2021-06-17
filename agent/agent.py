import numpy as np
import random
from collections import namedtuple, deque
from .Qnetworks import SimpleQNetwork as QNetwork
import torch
import torch.optim as optim
import os

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, parser):
        """Initialize an Agent object.
        s
        Params that we will use:
        ======
            env (environmet object): environment for the agent, see ./environments for more details
            parser (argparse): parser with all training flags (see main.py)
        """

        # save the environment
        self.env = env
        # saveroot for the model checkpoints
        self.savedir = os.path.join(parser.checkpoints_dir, parser.name)
        # how many actions can each agent do
        self.action_size = parser.action_size
        # how many agents we have
        self.n_agents = parser.n_agents
        # random seed for reproducibility
        self.seed = random.seed(parser.seed)
        # get the device for gpu dependencies
        self.device = torch.device('cuda' if parser.use_cuda else 'cpu')
        # discount factor
        self.gamma = parser.gamma
        # flag for hard/soft update
        self.target_update = parser.target_update
        # delay for hard update
        self.delay_steps = parser.delay_steps
        # tau for soft target network update
        self.tau = parser.tau
        # lr for q networks
        self.lr = parser.learning_rate
        # learn from buffer every ``update_every`` steps
        self.update_every = parser.update_every
        # purely exploring steps at the beginning
        self.exploring_steps = parser.exploring_steps
        # loss
        self.loss = torch.nn.SmoothL1Loss()
        # batch size
        self.batch_size = parser.batch_size
        # replay buffer size
        self.buffer_size = parser.buffer_size

        # Q-Network
        self.qnetwork_local = QNetwork((1, env.sx, env.sy), self.action_size, self.n_agents, parser.seed, parser.n_blocks_Q,
                                       parser.downsampling_Q, parser.n_features_Q, not parser.no_dropout_Q).to(self.device)
        print("Q Network instanciated: (%d parameters)"%self.qnetwork_local.count_parameters())
        print(self.qnetwork_local)
        self.qnetwork_target = QNetwork((1, env.sx, env.sy), self.action_size, self.n_agents, parser.seed, parser.n_blocks_Q,
                                        parser.downsampling_Q, parser.n_features_Q, not parser.no_dropout_Q).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, actions, reward, next_state):
        # convert increment actions back to their ids
        actions = np.vstack([self.mapIncrementToAction(a) for a in actions])
        # Save experience in replay memory
        self.memory.add(state, actions, reward, next_state)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step+= 1
        if self.t_step % self.update_every == 0 and self.t_step>self.exploring_steps and len(self.memory) > self.batch_size:
            # If enough samples are available in memory, get random subset and learn
            experiences = self.memory.sample()
            loss = self.learn(experiences)
            return loss
        else:
            return 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        slice = self.env.sample(state).unsqueeze(0).unsqueeze(0)/255 # normalize slice (which is a uint8) before inputiing it to the Qnetwork
        self.qnetwork_local.eval()
        with torch.no_grad():
            Qs = self.qnetwork_local(slice)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps and self.t_step>self.exploring_steps:
            return np.vstack([self.mapActionToIncrement(torch.argmax(Q, dim=1).item()) for Q in Qs])
        else:
            if self.t_step == self.exploring_steps:
                print("finished %d exploring steps, starting to train the agent every %d steps."%(self.exploring_steps, self.update_every))
            return np.vstack([self.mapActionToIncrement(random.choice(np.arange(self.action_size)))for _ in range(self.n_agents)])

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states = experiences

        # get corresponding slices for the states and next states (divide by 255 to normalize input as sample() yields uint8 slices)
        states = torch.cat([self.env.sample(state=s).unsqueeze(0).unsqueeze(0)/255 for s in states], axis=0).float().to(self.device)
        next_states = torch.cat([self.env.sample(state=s).unsqueeze(0).unsqueeze(0)/255 for s in next_states], axis=0).float().to(self.device)
        # convert rewards and actions to tensor and move to gpu
        rewards, actions = torch.from_numpy(rewards).float().to(self.device), torch.from_numpy(actions).long().to(self.device)
        # get the action values of the current states and the target values of the next states 
        Qs = self.qnetwork_local(states)
        MaxQs = self.qnetwork_target(next_states)
        # train the Qnetwork
        self.optimizer.zero_grad()
        for Q, A, MaxQ in zip(Qs, actions.permute(1,0,2), MaxQs):
            # gather Q value for the action taken
            Q = Q.gather(1, A)
            # get the value of the best action in the next state
            # detach to only optimize local network
            MaxQ = MaxQ.max(1)[0].detach().unsqueeze(-1)
            # backup the expected value of this action  
            Qhat = rewards + self.gamma*MaxQ
            # evalauate TD error
            loss = self.loss(Q, Qhat)
            # retain graph because we will backprop multiple times through the backbone cnn
            loss.backward(retain_graph=True)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)   
        elif self.target_update == "hard":
            self.hard_update(self.qnetwork_local, self.qnetwork_target, self.delay_steps)
        else:
            raise ValueError('unknown ``self.target_update``: {}. possible options: [hard, soft]'.format(self.target_update))

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
    
    def hard_update(self, local_model, target_model, N):
        """hard update model parameters.
        θ_target = θ_local every N steps.
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            N (flintoat): number of steps after which hard update takes place 
        """
        if self.t_step % N == 0:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)

    def save(self, fname="latest.pth"):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        torch.save(self.qnetwork_target.state_dict(), os.path.join(self.savedir, fname))

    def load(self, name):
        print("loading: {}".format(os.path.join(self.savedir, name)))
        if not ".pth" in name:
            name+=".pth"
        state_dict = torch.load(os.path.join(self.savedir, name))
        self.qnetwork_local.load_state_dict(state_dict)
        self.qnetwork_target.load_state_dict(state_dict)

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
        states = np.vstack([e.state[np.newaxis, ...] for e in experiences if e is not None])
        actions = np.vstack([e.action[np.newaxis, ...] for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state[np.newaxis, ...] for e in experiences if e is not None])
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)