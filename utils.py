
import torch, os, wandb
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from environment.xcatEnvironment import *
from agent.agent import Agent
from buffer.buffer import *
from visualisation.visualizers import Visualizer

# ==== THE FOLLOWING FUNCTIONS HANDLE TRAINING AND TESTING OF THE AGENTs ====
def train(config, local_model, target_model, wandb_entity="us_navigation", sweep=False, rank=0):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                config (argparse object): configuration with all training options. (see options/options.py)
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                                             (if more processes, the Qnetwork is shared)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                              (it is a hard copy or a running average of the local model, helps against diverging)
                wandb_entuty (str): which wandb workspace to save logs to. (if unsure use your main workspace i.e. your-user-name)
                sweep (bool): flag if we are performing a sweep, in which case we will not be saving checkpoints as that will occupy too much memory.
                              However we will still save the final model in .onnx format (only intermediate .pth checkpoints are not saved)
                rank (int): indicates the process number if multiple processes are queried
        """ 
        # ==== instanciate useful classes ====

        # manual seed
        torch.manual_seed(config.seed + rank) 
        # 1. instanciate environment
        env = setup_environment(config)
        # 2. instanciate agent
        agent = Agent(config)
        # 3. instanciate optimizer for local_network
        optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
        # 4. instanciate criterion
        criterion = setup_criterion(config)
        # 5. instanciate replay buffer
        buffer = PrioritizedReplayBuffer(config.buffer_size, config.batch_size, config.alpha)
        # 6. instanciate visualizer
        visualizer = Visualizer(agent.results_dir)

        # ==== LAUNCH TRAINING ====
        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                print("random walk to collect experience...")
                env.random_walk(config.exploring_steps, buffer)  
        # 2. initialize wandb for logging purposes
        if config.wandb in ["online", "offline"]:
                wandb.login()
        wandb.init(entity=wandb_entity, config=config, mode=config.wandb, name=config.name)
        config = wandb.config # oddly this ensures wandb works smoothly
        # 3. tell wandb to watch what the model gets up to: gradients, weights, and loss
        wandb.watch(local_model, criterion, log="all", log_freq=config.log_freq)
        # 4. start training
        for episode in tqdm(range(config.n_episodes), desc="training..."):
                logs = agent.play_episode(env, local_model, target_model, optimizer, criterion, buffer)
                # send logs to weights and biases
                if episode % config.log_freq == 0:
                        wandb.log(logs, commit=True)
                # save agent locally and test its current greedy policy
                if episode % config.save_freq == 0:
                        if not sweep:
                                print("saving latest model weights...")
                                local_model.save(os.path.join(agent.checkpoints_dir, "latest.pth"))
                                target_model.save(os.path.join(agent.checkpoints_dir, "episode%d.pth"%episode))
                        # test the greedy policy and send logs
                        out = agent.test_agent(config.n_steps_per_episode, env, local_model)
                        wandb.log(out["wandb"], commit=True)
                        # animate the trajectory followed by the agent in the current episode
                        visualizer.render_frames(out["frames"], "episode%d.gif"%episode)
                        # upload file to wandb
                        wandb.save(os.path.join(visualizer.savedir, "episode%d.gif"%episode))
        # at the end of the training session save the model as .onnx to improve the open sourceness and exchange-ability amongst different ML frameworks
        sample_inputs = torch.tensor(out["frames"][:agent.config.batch_size]) # if location aware this will be already of shape BxCxHxW otherwise this will be BxHxW.
        if len(sample_inputs.shape) == 3:
                sample_inputs = sample_inputs.unsqueeze(1)
        torch.onnx.export(local_model, sample_inputs.float().to(agent.config.device), os.path.join(agent.checkpoints_dir, "DQN.onnx"))
        # upload file to wandb
        wandb.save(os.path.join(agent.checkpoints_dir, "DQN.onnx"))

def setup_environment(config):
    if not config.location_aware and not config.CT2US:
        env = SingleVolumeEnvironment(config)
    elif not config.location_aware and config.CT2US: 
        env = CT2USSingleVolumeEnvironment(config)
    elif config.location_aware and not config.CT2US:
        env = LocationAwareSingleVolumeEnvironment(config)
    else:
        raise NotImplementedError()
    return env

def setup_criterion(config):
    if "mse" in config.loss.lower():
        criterion = nn.MSELoss(reduction='none')
    elif "smooth" in config.loss.lower():
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError()
    return criterion