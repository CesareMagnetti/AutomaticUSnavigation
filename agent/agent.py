
from agent.baseAgent import BaseAgent
import numpy as np
import torch

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
        # Initialize time step count
        self.t_step = 0
        # formulate a suitable decay factor for epsilon given the queried options.
        self.EPS_DECAY_FACTOR = (config.eps_end/config.eps_start)**(1/int(config.stop_eps_decay*config.n_episodes))

    def train(self, env):  
        eps = self.config.eps_start
        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.qnetwork_target, self.loss, log="all", log_freq=self.config.log_freq)
        for episode in range(self.config.n_episodes):
            # reset env to a random initial slice
            env.reset()
            slice = env.sample(env.state)
            logs = {log: 0 for log in env.logged_rewards}
            logs["TDerror"] = 0
            for _ in range(self.config.n_steps_per_episode):  
                # increase time step
                self.t_step+=1
                # get action from current state
                actions = self.act(slice, eps)  
                # observe next state (automatically adds (state, action, reward, next_state) to env.buffer) 
                next_slice, rewards = env.step(actions)
                # learn every UPDATE_EVERY steps, only after EXPLORING_STEPS and if enough samples in env.buffer
                if self.t_step % self.config.update_every == 0 and self.t_step>self.config.exploring_steps and len(env.buffer) > self.config.batch_size:
                    batch = self.memory.sample()
                    TDerror = self.learn(batch)
                else:
                    TDerror = 0
                # set slice to next slice
                slice= next_slice
                # add to logs
                logs["TDerror"]+=TDerror
                for r in rewards:
                    logs[r]+=rewards[r]
            # save logs to wandb
            if episode%self.config.log_freq == 0:
                wandb.log(logs, step=self.t_step, commit=True)
                    # save agent locally
            if episode % self.config.save_freq == 0 and self.t_step>self.config.exploring_steps:
                print("saving latest model weights...")
                self.save()
                self.save("episode%d.pth"%episode)
                # tests the agent greedily for logs
                self.test(env, "episode%d.gif"%episode)
            # update eps
            if self.t_step>self.config.exploring_steps:
                eps = max(eps*self.EPS_DECAY_FACTOR, self.config.eps_end)
        

    def learn(self, batch, env):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
            env (environment/* object): environment the agent is using to learn a good policy
        """
        # 1. organize batch
        states, actions, rewards, next_states = batch
        planes = env.sample_planes(states+next_states)
        # concatenate states, normalize and move to gpu
        states = torch.from_numpy(np.vstack(planes[:self.batch_size][np.newaxis, np.newaxis, ...]/255)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(planes[self.batch_size:][np.newaxis, np.newaxis, ...]/255)).float().to(self.device)
        print("states: ", states.shape, "next_states: ", next_states.shape)
        # convert rewards to tensor and move to gpu
        rewards = torch.from_numpy(rewards).float().unsqueeze(0).to(self.device)
        print("rewards: ", rewards.shape)
        # convert actions to tensor, actions are currently stored as a list of length batch_size, where each entry is
        # an np.vstack([action1, action2, action3]). We need to convert this to a list of:
        # [action1]*batch_size + [action2]*batch_size + [action3]*batch_size so thats it's coherent with Q and MaxQ
        print("actions from buffer: ", actions)
        actions = torch.from_numpy(np.hstack(actions)).long().to(self.device)
        print("actions rearranged: ", actions, actions.shape)
        # get the action values of the current states and the target values of the next states 
        # concatenate outputs of all agents so that we have shape: batch_sizex(n_agents*action_size)
        Q = torch.cat(self.qnetwork_local(states), dim=-1)
        MaxQ = torch.cat(self.qnetwork_target(next_states), dim=-1)
        print("Q: ", Q.shape, "MaxQ: ", MaxQ.shape)
        # train the Qnetwork
        # REMEMBER TO RESHAPE ACTIONS!!!
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