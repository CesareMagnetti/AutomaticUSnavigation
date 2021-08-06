
import torch, os, six, random
import numpy as np
from abc import abstractmethod, ABCMeta
from agent.trainers import *

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

        # setup the action size and the number of agents
        self.n_agents, self.action_size = config.n_agents, config.action_size

        # place holder for steps and episode counts
        self.t_step, self.episode = 0, 0
        # starting epsilon value for exploration/exploitation trade off
        self.eps = config.eps_start
        # formulate a suitable decay factor for epsilon given the queried options.
        self.EPS_DECAY_FACTOR = (config.eps_end/config.eps_start)**(1/int(config.stop_decay*config.n_episodes))
        # starting beta value for bias correction in prioritized experience replay
        self.beta = config.beta_start
        # formulate a suitable decay factor for beta given the queried options. (since beta_end>beta_start, this will actually be an increase factor)
        # annealiate beta to 1 (or beta_end) as we go further in the episode (original P.E.R paper reccommends this)
        self.BETA_DECAY_FACTOR = (config.beta_end/config.beta_start)**(1/int(config.stop_decay*config.n_episodes))
        # set the trainer algorithm
        if config.trainer.lower() in ["deepqlearning", "qlearning", "dqn"]:
            self.trainer = PrioritizedDeepQLearning(gamma=config.gamma)
        elif config.trainer.lower() in ["doubledeepqlearning", "doubleqlearning", "doubledqn", "ddqn"]:
            self.trainer = PrioritizedDoubleDeepQLearning(gamma=config.gamma)
        else:
            raise NotImplementedError('unknown ``trainer`` configuration: {}. available options: [DQN, DoubleDQN]'.format(config.trainer))

        # save the config for any options we might need
        self.config = config
    
    def random_action(self):
        """Return a random discrete action for each agent, stack them vertically.
        """
        return np.vstack([random.choice(np.arange(self.action_size)) for _ in range(self.n_agents)])
    
    def greedy_action(self, slice, local_model):
        """Returns the discrete actions for which the Q values of each agent are maximized, stacked vertically.
        Params:
        ==========
            slice (np.ndarray of shape (H, W)): 2D slice of anatomy.
            local_model (PyTorch model): takes input the slice and outputs action values.

        returns:
            np.ndarray (contains an action for each agent)
        """
        # convert to tensor, normalize and unsqueeze to pass through qnetwork
        slice = torch.from_numpy(slice).float().to(self.config.device)
        with torch.no_grad():
            Qs = local_model(slice)
        return np.vstack([torch.argmax(Q, dim=1).item() for Q in Qs])

    def act(self, plane, local_model, eps=0.):
        """Generate an action given some input.
        Params:
        ==========
            plane (np.ndarray of shape (H, W)): 2D slice of anatomy.
            local_model (PyTorch model): takes input the slice and outputs action values
            eps (float): epsilon parameter governing exploration/exploitation trade-off
        """
        if random.random()>eps:
            return self.greedy_action(plane, local_model)
        else:
            return self.random_action()
    
    @abstractmethod
    def learn(self):
        """update the Q network through some routine.
        """
    
    @abstractmethod
    def play_episode(self):
        """Make the agent play a full episode.
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