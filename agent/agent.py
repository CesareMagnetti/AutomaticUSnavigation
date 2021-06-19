import numpy as np
import random
from .Qnetworks import SimpleQNetwork as QNetwork
import torch
import torch.optim as optim
import os
from timer.timer import Timer
import concurrent.futures
import wandb

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, parser):
        """Initialize an Agent object.
        s
        Params that we will use:
        ======
            parser (argparse object): parser with all training options (see options/options.py)
        """
        # Initialize time step count
        self.t_step = 0
        # saveroot for the model checkpoints
        self.savedir = os.path.join(parser.checkpoints_dir, parser.name)
        # get the device for gpu dependencies
        self.device = torch.device('cuda' if parser.use_cuda else 'cpu')
        # all other needed configs
        self.config = parser
        # formulate a suitable decay factor for epsilon given the queried options.
        self.EPS_DECAY_FACTOR = (parser.eps_end/parser.eps_start)**(1/int(parser.stop_eps_decay*parser.n_episodes - parser.exploring_steps/parser.n_steps_per_episode))
        # loss
        self.loss = torch.nn.MSELoss()
        # Q-Network
        self.qnetwork_local = QNetwork((1, parser.load_size, parser.load_size), parser.action_size, parser.n_agents, parser.seed, parser.n_blocks_Q,
                                       parser.downsampling_Q, parser.n_features_Q, not parser.no_dropout_Q).to(self.device)
        self.qnetwork_target = QNetwork((1, parser.load_size, parser.load_size), parser.action_size, parser.n_agents, parser.seed, parser.n_blocks_Q,
                                        parser.downsampling_Q, parser.n_features_Q, not parser.no_dropout_Q).to(self.device)
        print("Q Network instanciated: (%d parameters)\nrun with --print_network flag to see network details"%self.qnetwork_local.count_parameters())
        if parser.print_network:
            print(self.qnetwork_local)
        # Optimizer for local network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=parser.learning_rate)
        

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
        
    def act(self, slice, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            slice (torch.tensor): a slice through the volume (see SingleVolumeEnvironment.sample())
            eps (float): epsilon, for epsilon-greedy action selection. By default it will return the greedy action.
        """
        self.qnetwork_local.eval()
        with torch.no_grad():
            Qs = self.qnetwork_local(slice)
        self.qnetwork_local.train()
        # greedy action with prob. (1-eps) and only if we are done exploring
        if random.random() > eps and self.t_step>self.exploring_steps:
            return np.vstack([torch.argmax(Q, dim=1).item() for Q in Qs])
        # random action otherwise
        else:
            return np.vstack([random.choice(np.arange(self.action_size)) for _ in range(self.n_agents)])

    def learn(self, batch):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states = batch
        # sample planes using multi-thread
        # planes = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.env.sample, s) for s in states+next_states]
        # [planes.append(f.result()) for f in futures]
        # sample planes using a single-thread
        planes = [self.env.sample(state=s) for s in states+next_states]
        # concatenate states and move to gpu
        states = torch.cat(planes[:self.batch_size], axis=0)
        next_states = torch.cat(planes[self.batch_size:], axis=0)
        print("states: ", states.shape, "next_states: ", next_states.shape)
        # convert rewards to tensor and move to gpu
        rewards = torch.from_numpy(rewards).float().to(self.device)
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