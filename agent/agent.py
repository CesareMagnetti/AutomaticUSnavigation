from agent.baseAgent import BaseAgent
import numpy as np
import torch
from tqdm import tqdm
import concurrent.futures
from collections import deque

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

    def play_episode(self, env, local_model, buffer):
        """ Plays one episode on an input environment.
        Params:
        ==========
            env (environment/* instance): the environment the agent will interact with while training.
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            buffer (buffer/* object): replay buffer shared amongst processes (each process pushes to the same memory.)
        Returns logs (dict): all relevant logs acquired throughout the episode.
        """  
        self.episode+=1
        env.reset()
        sample = env.sample_plane(env.state, preprocess=True)
        if self.config.recurrent:
            plane_history = deque(maxlen=self.config.recurrent_history_len)  # instanciate history at beginning of each episode
        # play episode (stores transition tuples to the buffer)
        with torch.no_grad():
            for i in range(self.config.n_steps_per_episode): 
                # if recurrent add plane to history
                if self.config.recurrent:
                    plane_history.append(sample["plane"]) 
                    # if less than config.recurrent_history_len planes, then pad to the left with oldest plane in history
                    if len(plane_history)<self.config.recurrent_history_len:
                        n_pad = self.config.recurrent_history_len - len(plane_history)
                        tensor_pad = plane_history[0]
                        # concatenate the history to 1*LxCxHxW
                        plane = np.concatenate([tensor_pad]*n_pad + list(plane_history), axis=0)
                    else:
                        plane = np.concatenate(list(plane_history), axis=0)
                # else the current plane is passed through the Qnetwork
                else:
                    plane = sample["plane"]
                self.t_step+=1
                # get action from current state
                actions = self.act(plane, local_model, self.eps) 
                # step the environment to return a transitiony  
                transition, next_sample = env.step(actions, preprocess=True)
                # add (state, action, reward, next_state) to buffer
                if self.config.recurrent: buffer.add(transition, is_first_time_step = i == 0)
                else: buffer.add(transition)
                # set sample to next sample
                sample= next_sample
                # if done, end episode early
                if transition[-1]:
                    break
        # return episode logs
        logs = env.logs
        logs.update({"epsilon": self.eps, "beta": self.beta})
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
        if self.config.recurrent:
            plane_history = deque(maxlen=self.config.recurrent_history_len)  # instanciate history at beginning of each episode
        # play an episode greedily
        with torch.no_grad():
            for _ in range(1, steps+1):
                # if recurrent add plane to history
                if self.config.recurrent:
                    plane_history.append(sample["plane"]) 
                    # if less than config.recurrent_history_len planes, then pad to the left with oldest plane in history
                    if len(plane_history)<self.config.recurrent_history_len:
                        n_pad = self.config.recurrent_history_len - len(plane_history)
                        tensor_pad = plane_history[0]
                        # concatenate the history to 1*LxCxHxW
                        plane = np.concatenate([tensor_pad]*n_pad + list(plane_history), axis=0)
                    else:
                        plane = np.concatenate(list(plane_history), axis=0)
                # else the current plane is passed through the Qnetwork
                else:
                    plane = sample["plane"]
                # get action from current state
                actions = self.act(plane, local_model)
                # observe transition and next_slice
                transition, next_sample = env.step(actions, preprocess=True)
                # set slice to next slice
                sample = next_sample
                # add logs to output dict  
                out["planes"].append(sample["plane"].squeeze())
                if not self.config.realCT:
                    out["segs"].append(sample["seg"].squeeze())
                if self.config.CT2US:
                    out["planesCT"].append(sample["planeCT"].squeeze())
                out["states"].append(env.state)
                # when we test on clinical data we do not know the rewards we get
                if not self.config.realCT:
                    #out["logs"].append({log: r for log,r in env.logs.items()}) # cumulative logs
                    out["logs"].append({log: r for log,r in env.current_logs.items()})
                # if done, end episode early and pad logs with terminal state
                if transition[-1]:
                    break
        # when we test on clinical data we do not know the rewards we get
        if not self.config.realCT:
            # add logs for wandb to out
            out["wandb"] = {log+"_test": r for log,r in env.logs.items()}
            # re-arrange the logs key so that it goes from list of dicts to dict of lists
            out["logs"] = {k: [dic[k] for dic in out["logs"]] for k in out["logs"][0]}
            # get the mean rewards collected by the agents in the episode
            out["logs_mean"] = {key: np.mean(val) for key, val in out["logs"].items()}
        return out
    
    def train(self, env, buffer, local_model, target_model, optimizer, criterion, n_iter=1):
        # train the q_network for a number of iterations
        local_model.train()
        total_loss = 0
        for i in range(n_iter):
            # 1. sample batch and send to GPU
            batch = self.prepare_batch(env, buffer)
            for key in batch:
                if key!="indices":
                    batch[key] = batch[key].to(self.config.device)
            # 2. take a training step
            loss, deltas =self.learn(batch, local_model, target_model, optimizer, criterion)
            # 3. update priorities
            buffer.update_priorities(batch["indices"], deltas.cpu().detach().numpy().squeeze())
            # 4. add to total loss
            total_loss+=loss.item()
        # set back to eval mode as we will only be training inside this function
        local_model.eval()
        # decrease eps
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
        weights = torch.from_numpy(weights).float().squeeze()
        states, actions, rewards, next_states, dones = batch
        # this is used when recurrent Q network is used, states and next_states will be B*L x 3 x 3
        L = self.config.recurrent_history_len if self.config.recurrent else 1
        # 2. sample planes and next_planes using multiple threads
        sample = env.sample_planes(states+next_states, preprocess=True)
        # 3. preprocess each item
        states = torch.from_numpy(np.vstack(sample["plane"][:self.config.batch_size*L])).float()
        next_states = torch.from_numpy(np.vstack(sample["plane"][self.config.batch_size*L:])).float()
        rewards = torch.from_numpy(np.hstack(rewards)).unsqueeze(-1).float()
        actions = torch.from_numpy(np.hstack(actions)).unsqueeze(-1).long()
        dones = torch.tensor(dones).unsqueeze(-1).bool()
        # organize batch and return
        batch = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones, "weights": weights, "indices": indices}
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
    def play_episode(self, envs, local_model, buffers):
        """ Plays one episode on an input environment.
        Params:
        ==========
            envs dict[environment/* instance]: dict of environments the agent will interact with while training.
            local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
            buffers dict[buffer/* object]: dict of replay buffers (one per environment, same keys)
        Returns logs (dict): all relevant logs acquired throughout the episode.
        """  
        # play episode in each environment using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {vol_id: executor.submit(super(MultiVolumeAgent, self).play_episode, envs[vol_id],
                                                                                           local_model,
                                                                                           buffers[vol_id]) for vol_id in envs}
        logs = {vol_id: f.result() for vol_id, f in futures.items()}
        return logs
 
    # rewrite the prepare batch
    def prepare_batch(self, envs, buffers):
        # 1. sample batches in parallel from all buffers
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {vol_id: executor.submit(super(MultiVolumeAgent, self).prepare_batch, envs[vol_id], buffers[vol_id]) for vol_id in envs}
        batches = [f.result() for f in futures.values()]
        # 2. concatenate states, actions, rewards, next_states, weights and indices from all buffers into a single batch
        states = torch.cat([batch["states"] for batch in batches], dim=0)
        actions = torch.cat([batch["actions"] for batch in batches], dim=1)
        rewards = torch.cat([batch["rewards"] for batch in batches], dim=1)
        next_states = torch.cat([batch["next_states"] for batch in batches], dim=0)
        dones = torch.cat([batch["dones"] for batch in batches], dim=0)
        weights = torch.cat([batch["weights"] for batch in batches], dim=0)
        indices = np.stack([batch["indices"] for batch in batches])
        # organize batch and return
        batch = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones, "weights": weights, "indices": indices}
        return batch
    
    # rewrite the train function
    def train(self, envs, local_model, target_model, optimizer, criterion, buffers, n_iter=1):
        # train the q_network for a number of iterations
        local_model.train()
        total_loss = 0
        for _ in tqdm(range(n_iter), desc="training Qnetwork..."):
            # 1. sample batch and send to GPU
            batch = self.prepare_batch(envs, buffers)
            for key in batch:
                if key!="indices":
                    batch[key] = batch[key].to(self.config.device)
            # 2. take a training step
            loss, deltas = self.learn(batch, local_model, target_model, optimizer, criterion)
            # 3. update priorities for each buffer separately (do this in parallel)
            deltas = deltas.cpu().detach().numpy().squeeze().reshape(self.n_envs, -1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(buffer.update_priorities, ind, delt) for buffer, ind, delt in zip(buffers.values(), batch["indices"], deltas)]
            # 4. add to total loss
            total_loss+=loss.item()
        # set back to eval mode as we will only be training inside this function
        local_model.eval()
        # decrease eps
        self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        self.beta = min(self.beta*self.BETA_DECAY_FACTOR, self.config.beta_end)
        return loss/n_iter

    # rewrite the test agent function
    def test_agent(self, steps, envs, local_model):
        """Test the greedy policy learned by the agent on some test environments and returns a dict with useful metrics/logs.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            envs dict[environment/* instance]: dict of environments.
            local_model (PyTorch model): pytorch network that will be tested.
        """
        # test agent on each environment using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {vol_id: executor.submit(super(MultiVolumeAgent, self).test_agent, steps, envs[vol_id], local_model) for vol_id in envs}
        logs = {vol_id: f.result() for vol_id, f in futures.items()}
        return logs