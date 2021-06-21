
from agent.baseAgent import BaseAgent
from agent.trainers import DeepQLearning
import numpy as np
import torch, os, wandb
from moviepy.editor import ImageSequenceClip
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
        # place holder for steps and episode counts
        self.t_step, self.episode = 0, 0
        # starting epsilon value for exploration/exploitation trade off
        self.eps = self.config.eps_start
        # set the trainer algorithm
        if config.trainer.lower() in ["deepqlearning", "qlearning", "dqn"]:
            self.trainer = DeepQLearning(gamma=config.gamma)
        else:
            raise NotImplementedError()

    def play_episode(self, env):
        self.episode+=1
        episode_loss = 0
        env.reset()
        slice = env.sample_plane(env.state)
        for _ in range(self.config.n_steps_per_episode):  
            self.t_step+=1
            # get action from current state
            actions = self.act(slice, self.eps) 
            # observe next state (automatically adds (state, action, reward, next_state) to env.buffer) 
            next_slice = env.step(actions)
            # learn every UPDATE_EVERY steps and if enough samples in env.buffer
            if self.t_step % self.config.update_every == 0 and len(env.buffer) > self.config.batch_size:
                episode_loss+=self.learn(env.buffer.sample(), env)
            # set slice to next slice
            slice= next_slice

        # send logs to weights and biases
        if self.episode % self.config.log_freq == 0:
            logs = env.logs
            logs["loss"] = episode_loss
            logs["epsilon"] = self.eps
            wandb.log(logs, step=self.t_step, commit=True)

        # save agent locally and test its current greedy policy
        if self.episode % self.config.save_freq == 0:
            print("saving latest model weights...")
            self.save()
            self.save("episode%d.pth"%self.episode)
            # test current greedy policy
            self.test(self.config.n_steps_per_episode, env, "episode%d"%self.episode)

        # update eps
        if self.t_step>self.config.exploring_steps:
            self.eps = max(self.eps*self.EPS_DECAY_FACTOR, self.config.eps_end)

    def train(self, env):
        """ Trains the agent on an input environment.
        Params:
        ==========
            env (environment/* instance OR list[envs]): the environment the agent will interact with while training. 
                                                        if a list of environments is passed, they will run in parallel.
        """        
        # 1. initialize wandb for logging purposes
        if self.config.wandb in ["online", "offline"]:
            wandb.login()
        wandb.init(project="AutomaticUSnavigation", name=self.config.name, config=self.config, mode=self.config.wandb)
        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.qnetwork_target, self.loss, log="all", log_freq=self.config.log_freq)

        # 2. launch exploring steps if needed
        if self.config.exploring_steps>0:
            env.random_walk(self.config.exploring_steps, self.config.exploring_restarts)

        # 3. once we are done exploring launch training
        # LAUNCHES MANY ENVIRONMENTS IN PARALLEL
        if isinstance(env, (list, tuple)):
            for _ in tqdm(range(int(self.config.n_episodes/len(env))), desc="training..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    [executor.submit(self.play_episode(e)) for e in env]
        # LAUNCHES A SINGLE ENVIRONMENT SEQUENTIALLY
        else:
            for _ in tqdm(range(self.config.n_episodes), desc="training..."):
                self.play_episode(env)     
        # close wandb
        wandb.finish()

    def test(self, steps, env, fname):
        """Test the greedy policy learned by the agent, saves the trajectory as a GIF and logs collected reward to wandb.
        Params:
        ==========
            steps (int): number of steps to test the agent for.
            env (environment/* instance): the environment the agent will interact with while testing.
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
            actions = self.act(slice)  
            # observe next state (automatically adds (state, action, reward, next_state) to env.buffer) 
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
    
    def learn(self, batch, env):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
            env (environment/* object): environment the agent is using to learn a good policy
        """
        # 1. organize batch
        states, actions, rewards, next_states = batch
        planes = env.sample_planes(states+next_states, process=True)
        # concatenate and move to gpu
        states = torch.from_numpy(np.vstack(planes[:self.config.batch_size])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(planes[self.config.batch_size:])).float().to(self.device)
        rewards = torch.tensor(rewards).repeat(self.n_agents).float().to(self.device) # repeat for n_agents that share this reward
        actions = torch.from_numpy(np.hstack(actions)).flatten().unsqueeze(0).long().to(self.device) # unroll for each agent one by one
        batch = (states, actions, rewards, next_states)

        # 2. make a training step (retain graph because we will backprop multiple times through the backbone cnn)
        self.optimizer.zero_grad()
        loss = self.trainer.step(batch, self.qnetwork_local, self.qnetwork_target, self.loss)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # 3. update target network
        if self.config.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)   
        elif self.config.target_update == "hard":
            self.hard_update(self.qnetwork_local, self.qnetwork_target, self.config.delay_steps)
        else:
            raise ValueError('unknown ``self.target_update``: {}. possible options: [hard, soft]'.format(self.config.target_update))   

        return loss.item()              