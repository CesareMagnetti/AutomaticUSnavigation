
from agent.baseAgent import BaseAgent
from agent.trainers import DeepQLearning, DoubleDeepQLearning
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
        # set the trainer algorithm
        if config.trainer.lower() in ["deepqlearning", "qlearning", "dqn"]:
            self.trainer = DeepQLearning(gamma=config.gamma)
        elif config.trainer.lower() in ["doubledeepqlearning", "doubleqlearning", "doubledqn", "ddqn"]:
            self.trainer = DoubleDeepQLearning(gamma=config.gamma)
        else:
            raise NotImplementedError()

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
            # observe next state (automatically adds (state, action, reward, next_state) to env.buffer) 
            next_slice = env.step(actions, buffer)
            # learn every UPDATE_EVERY steps and if enough samples in env.buffer
            if self.t_step % self.config.update_every == 0 and len(buffer) > self.config.batch_size:
                episode_loss+=self.learn(env, buffer, local_model, target_model, optimizer, criterion)
            # set slice to next slice
            slice= next_slice
        # return episode logs
        logs = env.logs
        logs["loss"] = episode_loss
        logs["epsilon"] = self.eps
        # decrease eps
        self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        return logs

    def test_agent(self, steps, env, local_model, fname="test"):
        """Test the greedy policy learned by the agent, saves the trajectory as a GIF and logs collected reward to wandb.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            env (environment/* instance): the environment the agent will interact with while testing.
            local_model (PyTorch model): pytorch network that will be tested.
            fname (str): name of file to save (default = test)
        """
        # reset env to a random initial slice
        env.reset()
        slice = env.sample_plane(env.state)
        slices = []
        # play an episode greedily
        for _ in tqdm(range(1, steps+1), desc="testing..."):  
            # save frame
            slices.append(slice[..., np.newaxis]*np.ones(3))
            # get action from current state
            actions = self.act(slice, local_model)  
            # observe next state (we do not pass a buffer at test time)
            next_slice = env.step(actions)
            # set slice to next slice
            slice = next_slice
        
        # send logs to wandb and save trajectory
        wandb.log({log+"_test": r for log,r in env.logs.items()}, commit=True)
        clip = ImageSequenceClip(slices, fps=10)
        if not os.path.exists(os.path.join(self.results_dir, "visuals")):
            os.makedirs(os.path.join(self.results_dir, "visuals"))
        clip.write_gif(os.path.join(self.results_dir, "visuals", fname+".gif"), fps=10)
        wandb.save(os.path.join(self.results_dir, "visuals", fname+".gif"))
    
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
        states, actions, rewards, next_states = buffer.sample()
        planes = env.sample_planes(states+next_states, process=True)
        # concatenate and move to gpu
        states = torch.from_numpy(np.vstack(planes[:self.config.batch_size])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack(planes[self.config.batch_size:])).float().to(self.config.device)
        rewards = torch.tensor(rewards).float().to(self.config.device)
        actions = torch.from_numpy(np.hstack(actions)).unsqueeze(-1).long().to(self.config.device)
        batch = (states, actions, rewards, next_states)

        # 2. make a training step (retain graph because we will backprop multiple times through the backbone cnn)
        optimizer.zero_grad()
        loss = self.trainer.step(batch, local_model, target_model, criterion)
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