from agent.baseAgent import BaseAgent
import numpy as np
import torch, os, wandb
from tqdm import tqdm
import concurrent.futures

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
        # play episode (stores transition tuples to the buffer)
        with torch.no_grad():
            for _ in range(self.config.n_steps_per_episode):  
                self.t_step+=1
                # get action from current state
                actions = self.act(sample["plane"], local_model, self.eps) 
                # step the environment to return a transitiony  
                transition, next_sample = env.step(actions, preprocess=True)
                # add (state, action, reward, next_state) to buffer
                buffer.add(*transition)
                # learn every UPDATE_EVERY steps and if enough samples in env.buffer
                # if self.t_step % self.config.update_every == 0 and len(buffer) > self.config.batch_size:
                #     episode_loss+=self.learn(env, buffer, local_model, target_model, optimizer, criterion)
                # set sample to next sample
                sample= next_sample
        # return episode logs
        logs = env.logs
        #logs.update({"loss": episode_loss, "epsilon": self.eps, "beta": self.beta})
        logs.update({"epsilon": self.eps, "beta": self.beta})
        # # decrease eps
        # self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        # self.beta = min(self.beta*self.BETA_DECAY_FACTOR, self.config.beta_end)
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
        # re-arrange the logs key so that it goes from list of dicts to dict of lists
        out["logs"] = {k: [dic[k] for dic in out["logs"]] for k in out["logs"][0]}
        return out
    
    def train(self, env, buffer, local_model, target_model, optimizer, criterion, n_iter=1):
        # train the q_network for a number of iterations
        local_model.train()
        total_loss = 0
        for i in range(n_iter):
            # 1. sample batch
            batch = self.prepare_batch(env, buffer)
            # 2. take a training step
            loss, deltas =self.learn(batch, local_model, target_model, optimizer, criterion)
            # 3. update priorities
            buffer.update_priorities(batch["indices"], deltas.cpu().detach().numpy().squeeze())
            # 4. add to total loss
            total_loss+=loss.item()
        # set back to eval mode as we will only be training inside this function
        local_model.eval()
        # decrease eps and increase beta after each training step
        self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        self.beta = min(self.beta*self.BETA_DECAY_FACTOR, self.config.beta_end)
        return loss/n_iter

    def prepare_batch(self, env, buffer):
        """arranges a training batch for the learn function
        Params:
        ==========
            env (environment/* instance): the environment the agent will interact with while training.
            buffer (buffer/* object): replay buffer shared amongst processes (each process pushes to the same memory.)
        Returns: batch (tuple): preprocessed transitions
        """
        # 1. sample transitions, weights and indices (for prioritization)
        batch, weights, indices = buffer.sample(beta=self.beta)
        weights = torch.from_numpy(weights).float().squeeze().to(self.config.device)
        states, actions, rewards, next_states = batch
        # 2. sample planes and next_planes using multiple threads
        sample = env.sample_planes(states+next_states, preprocess=True)
        # 3. preprocess each item
        states = torch.from_numpy(np.vstack(sample["plane"][:self.config.batch_size])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack(sample["plane"][self.config.batch_size:])).float().to(self.config.device)
        rewards = torch.from_numpy(np.hstack(rewards)).unsqueeze(-1).float().to(self.config.device)
        actions = torch.from_numpy(np.hstack(actions)).unsqueeze(-1).long().to(self.config.device)
        # organize batch and return
        batch = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "weights": weights, "indices": indices}
        return batch

    def learn(self, batch, local_model, target_model, optimizer, criterion):
        """ Update value parameters using given batch of experience tuples.
        Params:
        ==========
            batch (dict): contains all training inputs (see self.prepare_batch())
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                          (it is a hard copy or a running average of the local model, helps against diverging)
            optimizer (PyTorch optimizer): optimizer to update the local network weights.
            criterion (PyTorch Module): loss to minimize in order to train the local network.
        """  
        # 1. take a training step (retain graph because we will backprop multiple times through the backbone cnn)
        optimizer.zero_grad()
        loss, deltas = self.trainer.step(batch, local_model, target_model, criterion)
        loss.backward(retain_graph=True)
        optimizer.step()
        # 2. update target network
        if self.config.target_update.lower() == "soft":
            self.soft_update(local_model, target_model, self.config.tau)   
        elif self.config.target_update.lower() == "hard":
            self.hard_update(local_model, target_model, self.config.delay_steps)
        else:
            raise ValueError('unknown ``self.target_update``: {}. possible options: [hard, soft]'.format(self.config.target_update))   
        return loss, deltas              


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
        # play episode in each environment using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {self.config.volume_ids.split(',')[idx]: executor.submit(super(MultiVolumeAgent, self).play_episode,
                                                                               env,
                                                                               local_model,
                                                                               target_model,
                                                                               optimizer,
                                                                               criterion,
                                                                               buffer) for idx, (env, buffer) in enumerate(zip(envs, buffers))}
        logs = {key: f.result() for key, f in futures.items()}
        return logs
 
    # rewrite the prepare batch
    def prepare_batch(self, envs, buffers):
        # 1. sample batches in parallel from all buffers
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {self.config.volume_ids.split(',')[idx]: executor.submit(super(MultiVolumeAgent, self).prepare_batch,
                                                                               env,
                                                                               buffer) for idx, (env, buffer) in enumerate(zip(envs, buffers))}
        batches = [f.result() for f in futures.values()]
        # 2. concatenate states, actions, rewards, next_states, weights and indices from all buffers into a single batch
        states = torch.cat([batch["states"] for batch in batches], dim=0).to(self.config.device)
        actions = torch.cat([batch["actions"] for batch in batches], dim=1).to(self.config.device)
        rewards = torch.cat([batch["rewards"] for batch in batches], dim=1).to(self.config.device)
        next_states = torch.cat([batch["next_states"] for batch in batches], dim=0).to(self.config.device)
        weights = torch.cat([batch["weights"] for batch in batches], dim=0).to(self.config.device)
        indices = np.stack([batch["indices"] for batch in batches])
        # organize batch and return
        batch = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "weights": weights, "indices": indices}
        return batch
    
    # rewrite the train function
    def train(self, envs, local_model, target_model, optimizer, criterion, buffers, n_iter=1):
        # train the q_network for a number of iterations
        local_model.train()
        total_loss = 0
        for i in tqdm(range(n_iter), desc="training Qnetwork..."):
            # 1. sample batch
            batch = self.prepare_batch(envs, buffers)
            # 2. take a training step
            loss, deltas = self.learn(batch, local_model, target_model, optimizer, criterion)
            # 3. update priorities for each buffer separately (do this in parallel)
            deltas = deltas.cpu().detach().numpy().squeeze().reshape(self.n_envs, -1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(buffer.update_priorities, ind, delt) for buffer, ind, delt in zip(buffers, batch["indices"], deltas)]
            # 4. add to total loss
            total_loss+=loss.item()
        # set back to eval mode as we will only be training inside this function
        local_model.eval()
        # decrease eps and increase beta after each training step
        self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        self.beta = min(self.beta*self.BETA_DECAY_FACTOR, self.config.beta_end)
        return loss/n_iter

    # rewrite the test agent function
    def test_agent(self, steps, envs, local_model):
        """Test the greedy policy learned by the agent on some test environments and returns a dict with useful metrics/logs.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            envs list[environment/* instance]: list of environments.
            local_model (PyTorch model): pytorch network that will be tested.
        """
        # test agent on each environment using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {self.config.volume_ids.split(',')[idx]: executor.submit(super(MultiVolumeAgent, self).test_agent,
                                                                               steps,
                                                                               env,
                                                                               local_model) for idx, env in enumerate(envs)}
        logs = {key: f.result() for key, f in futures.items()}
        return logs