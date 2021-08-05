from agent.baseAgent import BaseAgent
import numpy as np
import torch, os, wandb
from tqdm import tqdm

class SingleVolumeAgent(BaseAgent):
    """Interacts with and learns from a single environment volume."""
    def __init__(self, config):
        """Initialize an Agent object.
        Params:
        ======
            config (argparse object): parser with all training options (see options/options.py)
        """
        # Initialize the base class
        BaseAgent.__init__(self, config)        


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
        sample = env.sample_plane(env.state, preprocess=True)
        for _ in range(self.config.n_steps_per_episode):  
            self.t_step+=1
            # get action from current state
            actions = self.act(sample["plane"], local_model, self.eps) 
            # step the environment to return a transitiony  
            transition, next_sample = env.step(actions, preprocess=True)
            # add (state, action, reward, next_state) to buffer
            buffer.add(*transition)
            # learn every UPDATE_EVERY steps and if enough samples in env.buffer
            if self.t_step % self.config.update_every == 0 and len(buffer) > self.config.batch_size:
                episode_loss+=self.learn(env, buffer, local_model, target_model, optimizer, criterion)
            # set sample to next sample
            sample= next_sample
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
        out = {"planes": [], "segs": [], "states": [], "logs": []}
        if self.config.CT2US:
            out.update({"planesCT": []})
        # reset env to a random initial slice
        env.reset()
        sample = env.sample_plane(env.state, preprocess=True, return_seg=True)
        # play an episode greedily
        with torch.no_grad():
            for _ in tqdm(range(1, steps+1), desc="testing..."):
                # add logs to output dict  
                out["planes"].append(sample["plane"].squeeze())
                out["segs"].append(sample["seg"].squeeze())
                if self.config.CT2US:
                    out["planesCT"].append(sample["planeCT"].squeeze())
                out["states"].append(env.state)
                out["logs"].append({log: r for log,r in env.current_logs.items()})
                # get action from current state
                actions = self.act(sample["plane"], local_model)
                # observe transition and next_slice
                transition, next_sample = env.step(actions, preprocess=True, return_seg=True)
                # set slice to next slice
                sample = next_sample
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
        sample = env.sample_planes(states+next_states, preprocess=True)
        # concatenate and move to gpu
        states = torch.from_numpy(np.vstack(sample["planes"][:self.config.batch_size])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack(sample["planes"][self.config.batch_size:])).float().to(self.config.device)
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


class MultiVolumeAgent(SingleVolumeAgent):
    """Interacts with and learns from multiple environment volumes."""
    def __init__(self, config):
        """Initialize an Agent object.
        Params:
        ======
            config (argparse object): parser with all training options (see options/options.py)
        """
        # Initialize the base class
        SingleVolumeAgent.__init__(self, config)  
        # number of environments that we are training on
        self.n_envs = len(config.volume_ids.split(','))
        # initialize the environment we are using
        self.switch_env()
    
    def switch_env(self, env_id=None):
        if env_id:
            self.env_id = env_id
        else:
            self.env_id = np.random.randint(self.n_envs)
    
    # rewrite the play episode function
    def play_episode(self, envs, local_model, target_model, optimizer, criterion, buffers, env_id=None):
        """ Plays one episode on an input environment.
        Params:
        ==========
            envs list[environment/* instance]: list of environments the agent will interact with while training.
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                          (it is a hard copy or a running average of the local model, helps against diverging)
            optimizer (PyTorch optimizer): optimizer to update the local network weights.
            criterion (PyTorch Module): loss to minimize in order to train the local network.
            buffers list[buffer/* object]: list of replay buffers (one per environment)
            env_id (int, optional): which environment to use, if None chose at random.
        Returns logs (dict): all relevant logs acquired throughout the episode.
        """  
        # set the current environment and buffers used
        self.switch_env(env_id)
        env, buffer = envs[self.env_id], buffers[self.env_id]
        # call the parent play_episode class to play an episode with these environment and buffer
        logs = super().play_episode(env, local_model, target_model, optimizer, criterion, buffer)
        logs["env_id"] = self.env_id
        return logs
    
    # rewrite the test agent function
    def test_agent(self, steps, envs, local_model):
        """Test the greedy policy learned by the agent on some test environments and returns a dict with useful metrics/logs.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            envs list[environment/* instance] or environment/* instance: list of environments or a single environment.
            local_model (PyTorch model): pytorch network that will be tested.
        """
        if not isinstance(envs, (list, tuple)):
            envs = [envs] 
        logs = {}
        # test all queried envs   
        for idx, env in enumerate(envs):
            print("testing env: {} ([{}]/[{}]) ...".format(self.config.volume_ids.split(',')[idx], idx+1, len(envs)))
            logs[self.config.volume_ids.split(',')[idx]] = super().test_agent(steps, env, local_model)
        return logs