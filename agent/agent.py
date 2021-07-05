
from agent.baseAgent import BaseAgent
from agent.trainers import *
import numpy as np
import torch, os, wandb
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

class Agent(BaseAgent):
    """Interacts with and learns from a single environment volume."""
    def __init__(self, config):
        """Initialize an Agent object.
        Params:
        ======
            config (argparse object): parser with all training options (see options/options.py)
        """
        # Initialize the base class
        BaseAgent.__init__(self, config)        
        # place holder for steps and episode counts
        self.t_step, self.episode = 0, 0
        # starting epsilon value for exploration/exploitation trade off
        self.eps = self.config.eps_start
        # formulate a suitable decay factor for epsilon given the queried options.
        self.EPS_DECAY_FACTOR = (config.eps_end/config.eps_start)**(1/int(config.stop_decay*config.n_episodes))
        # starting beta value for bias correction in prioritized experience replay
        self.beta = self.config.beta_start
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

    def play_episode(self, env, local_model, target_model, optimizer, criterion, buffer):
        """ Plays one episode on an input environment.
        Params:
        ==========
            env (environment/* instance): the environment the agent will interact with while training.
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                          (it is a hard copy or a running average of the local model, helps against diverging)
            optimizer (PyTorch optimizer): optimizer to update the local network weights.
            criterion (PyTorch Module): loss to minimize in order to train the local network.
            buffer (buffer/* object): replay buffer shared amongst processes (each process pushes to the same memory.)
        Returns logs (dict): all relevant logs acquired throughout the episode.
        """  
        self.episode+=1
        episode_loss = 0
        env.reset()
        slice = env.sample_plane(env.state)
        for _ in range(self.config.n_steps_per_episode):  
            self.t_step+=1
            # get action from current state
            actions = self.act(slice, local_model, self.eps) 
            # step the environment to return a transitiony  
            transition, next_slice = env.step(actions)
            # add (state, action, reward, next_state) to buffer
            buffer.add(*transition)
            # learn every UPDATE_EVERY steps and if enough samples in env.buffer
            if self.t_step % self.config.update_every == 0 and len(buffer) > self.config.batch_size:
                episode_loss+=self.learn(env, buffer, local_model, target_model, optimizer, criterion)
            # set slice to next slice
            slice= next_slice
        # return episode logs
        logs = env.logs
        logs.update({"loss": episode_loss, "epsilon": self.eps, "beta": self.beta})
        # decrease eps
        self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        self.beta = min(self.beta*self.BETA_DECAY_FACTOR, self.config.beta_end)
        return logs

    def test_agent(self, steps, env, local_model):
        """Test the greedy policy learned by the agent and returns a dict with useful metrics/logs.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            env (environment/* instance): the environment the agent will interact with while testing.
            local_model (PyTorch model): pytorch network that will be tested.
        """
        out = {"frames": [], "states": [], "logs": []}
        # reset env to a random initial slice
        env.reset()
        frame = env.sample_plane(env.state)
        # play an episode greedily
        for _ in tqdm(range(1, steps+1), desc="testing..."):
            # add to output dict  
            out["frames"].append(frame)
            out["states"].append(env.state)
            out["logs"].append(env.current_logs)
            # get action from current state
            actions = self.act(frame, local_model)  
            # observe transition and next_slice
            transition, next_frame = env.step(actions)
            # set slice to next slice
            frame = next_frame
        # add logs for wandb to out
        out["wandb"] = {log+"_test": r for log,r in env.logs.items()}
        return out
        
    def learn(self, env, buffer, local_model, target_model, optimizer, criterion):
        """ Update value parameters using given batch of experience tuples.
        Params:
        ==========
            env (environment/* instance): the environment the agent will interact with while training.
            buffer (buffer/* object): replay buffer shared amongst processes (each process pushes to the same memory.)
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                          (it is a hard copy or a running average of the local model, helps against diverging)
            optimizer (PyTorch optimizer): optimizer to update the local network weights.
            criterion (PyTorch Module): loss to minimize in order to train the local network.
        """  
        # 1. organize batch
        batch, weights, indices = buffer.sample(beta=self.beta)
        weights = torch.from_numpy(weights).float().squeeze().to(self.config.device)
        states, actions, rewards, next_states = batch
        planes = env.sample_planes(states+next_states, process=True)
        # concatenate and move to gpu
        states = torch.from_numpy(np.vstack(planes[:self.config.batch_size])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack(planes[self.config.batch_size:])).float().to(self.config.device)
        rewards = torch.from_numpy(np.hstack(rewards)).unsqueeze(-1).float().to(self.config.device)
        actions = torch.from_numpy(np.hstack(actions)).unsqueeze(-1).long().to(self.config.device)
        batch = (states, actions, rewards, next_states)
        # 2. make a training step (retain graph because we will backprop multiple times through the backbone cnn)
        optimizer.zero_grad()
        loss = self.trainer.step(batch, local_model, target_model, criterion, buffer, weights, indices)
        loss.backward(retain_graph=True)
        optimizer.step()
        # 3. update target network
        if self.config.target_update.lower() == "soft":
            self.soft_update(local_model, target_model, self.config.tau)   
        elif self.config.target_update.lower() == "hard":
            self.hard_update(local_model, target_model, self.config.delay_steps)
        else:
            raise ValueError('unknown ``self.target_update``: {}. possible options: [hard, soft]'.format(self.config.target_update))   
        return loss.item()              