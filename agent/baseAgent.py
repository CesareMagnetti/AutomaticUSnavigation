
from agent.Qnetworks import SimpleQNetwork as QNetwork
import torch, os, six, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod, ABCMeta

@six.add_metaclass(ABCMeta)
class BaseAgent(object):
    """
    base class for all of our navigation environments objects
    """
    def __init__(self, config):
        """ Initialize environment object
        Params:
        =========
            config (argparse object): all useful options to build the environment (see options/options.py for details)
        """

        # setup checkpoints and results dirs for any logging/ input output
        self.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
        self.results_dir = os.path.join(config.results_dir, config.name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        # set up the device
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        # setup the action size and the number of agents
        self.n_agents, self.action_size = config.action_size, config.n_agents
        # setup the qnetworks
        self.qnetwork_local = QNetwork((1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, config.n_blocks_Q,
                                       config.downsampling_Q, config.n_features_Q, not config.no_dropout_Q).to(self.device)
        self.qnetwork_target = QNetwork((1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, config.n_blocks_Q,
                                        config.downsampling_Q, config.n_features_Q, not config.no_dropout_Q).to(self.device)
        # setup the optimizer for the local qnetwork
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.learning_rate)
        # setup the training loss
        if "mse" in config.loss.lower():
            self.loss = nn.MSELoss()
        elif "smooth" in config.loss.lower():
            self.loss = nn.SmoothL1Loss()

        # save the config for any options we might need
        self.config = config
    
    def random_action(self):
        """Return a random discrete action for each agent, stack them vertically.
        """
        return np.vstack([random.choice(np.arange(self.action_size)) for _ in range(self.n_agents)])
    
    def greedy_action(self, slice):
        """Returns the discrete actions for which the Q values of each agent are maximized, stacked vertically.
        Params:
        ==========
            slice (np.ndarray of shape (H, W)): 2D slice of anatomy.
        """
        # convert to tensor, normalize and unsqueeze to pass through qnetwork
        slice = torch.from_numpy(slice/255, device = self.device).float().unsqueeze(0).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            Qs = self.qnetwork_local(slice)
        self.qnetwork_local.train()
        return np.vstack([torch.argmax(Q, dim=1).item() for Q in Qs])

    def act(self, slice, eps=0.):
        """Generate an action given some input.
        Params:
        ==========
            slice (np.ndarray of shape (H, W)): 2D slice of anatomy.
            eps (float): epsilon parameter governing exploration/exploitation trade-off
        """
        if random.sample()>eps:
            return self.greedy_action(slice)
        else:
            return self.random_action()
    
    @abstractmethod
    def learn(self):
        """update the Q network through some routine.
        """

    def train(self):
        raise NotImplementedError()

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
        print("saving: {}".format(os.path.join(self.checkpoints_dir, fname)))
        torch.save(self.qnetwork_target.state_dict(), os.path.join(self.checkpoints_dir, fname))

    def load(self, name):
        if not ".pth" in name:
            name+=".pth"
        print("loading: {}".format(os.path.join(self.checkpoints_dir, name)))
        state_dict = torch.load(os.path.join(self.checkpoints_dir, name))
        self.qnetwork_local.load_state_dict(state_dict)
        self.qnetwork_target.load_state_dict(state_dict)