
import torch, os, wandb
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from environment.xcatEnvironment import *
from environment.CT2USenvironment import *
from agent.agent import MultiVolumeAgent
from buffer.buffer import *
from visualisation.visualizers import Visualizer

# ==== THE FOLLOWING FUNCTIONS HANDLE TRAINING AND TESTING OF THE AGENTs ====
def train(config, local_model, target_model, name, wandb_entity="us_navigation", sweep=False):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                config (argparse object): configuration with all training options. (see options/options.py)
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                                             (if more processes, the Qnetwork is shared)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                              (it is a hard copy or a running average of the local model, helps against diverging)
                name (str): experiments name
                wandb_entuty (str): which wandb workspace to save logs to. (if unsure use your main workspace i.e. your-user-name)
                sweep (bool): flag if we are performing a sweep, in which case we will not be saving checkpoints as that will occupy too much memory.
                              However we will still save the final model in .onnx format (only intermediate .pth checkpoints are not saved)
        """ 
        # ==== instanciate useful classes ==== 
        # 1. instanciate environment(s)
        envs = setup_environment(config)
        # 2. instanciate agent
        agent = MultiVolumeAgent(config)
        # 3. instanciate optimizer for local_network
        optimizer = setup_optimizer(config, local_model)
        # 4. instanciate criterion
        criterion = setup_criterion(config)
        # 5. instanciate replay buffer(s) (one per environment)
        buffers = setup_buffers(config)
        # 6. instanciate visualizer
        visualizer = Visualizer(agent.results_dir)

        # ==== LAUNCH TRAINING ====
        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                for idx, (env,buffer) in enumerate(zip(envs.values(), buffers.values()), 1):
                    print("[{}]/[{}] random walk to collect experience...".format(idx, len(envs)))
                    env.random_walk(int(config.exploring_steps/len(envs)), buffer)  
        # 2. initialize wandb for logging purposes
        if config.wandb in ["online", "offline"]:
                wandb.login()
        wandb.init(entity=wandb_entity, config=config, mode=config.wandb, name=config.name, settings=wandb.Settings(start_method="fork"))
        config = wandb.config # oddly this ensures wandb works smoothly
        # 3. tell wandb to watch what the model gets up to: gradients, weights, and loss
        wandb.watch(local_model, criterion, log="all", log_freq=config.log_freq)
        # 4. start training
        for episode in tqdm(range(config.starting_episode+1, config.n_episodes+1), desc="playing episode..."):
                logs = agent.play_episode(envs, local_model, buffers)
                logs["loss"] = agent.train(envs, local_model, target_model, optimizer, criterion, buffers,
                                           n_iter = int(config.n_steps_per_episode/config.update_every))
                # send logs to weights and biases
                if episode % max(1, int(config.log_freq/len(envs))) == 0:
                        wandb.log(logs, commit=True)
                # save agent locally and test its current greedy policy
                local_model.save(os.path.join(agent.checkpoints_dir, "latest.pth"))
                torch.save(optimizer.state_dict(), os.path.join(agent.checkpoints_dir, "latest_optimizer.pth"))
                for vol_id, buffer in buffers.items():
                    buffer.save(os.path.join(config.checkpoints_dir, config.name, "latest_{}_".format(vol_id)))
                if episode % max(1, int(config.save_freq)) == 0 or episode == 1:
                        if not sweep:
                            print("saving model, optimizer and buffer...")
                            local_model.save(os.path.join(agent.checkpoints_dir, "episode%d.pth"%episode))
                            torch.save(optimizer.state_dict(), os.path.join(agent.checkpoints_dir, "episode%d_optimizer.pth"%episode))
                            for vol_id, buffer in buffers.items():
                                buffer.save(os.path.join(config.checkpoints_dir, config.name, "episode{}_{}_".format(episode, vol_id)))
                        # test the greedy policy on a random environment and send logs to wandb
                        test_env_id = np.random.choice(config.volume_ids.split(","))
                        out = agent.test_agent(config.n_steps_per_episode, {test_env_id: envs[test_env_id]}, local_model)
                        for _, log in out.items():
                            wandb.log(log["wandb"], commit=True)
                            # animate the trajectory followed by the agent in the current episode
                            if agent.config.CT2US:
                                visualizer.render_frames(log["planes"], log["planesCT"], n_rows = 2 if agent.config.location_aware else 1, fname = "episode%d.gif"%episode)
                            else:
                                visualizer.render_frames(log["planes"], n_rows = 2 if agent.config.location_aware else 1, fname = "episode%d.gif"%episode)
                        # upload file to wandb
                        wandb.save(os.path.join(visualizer.savedir, "episode%d.gif"%episode))
        # at the end of the training session save the model as .onnx to improve the open sourceness and exchange-ability amongst different ML frameworks
        nchannels = 1 if not config.location_aware else 4
        sample_inputs = torch.rand(agent.config.batch_size, nchannels, agent.config.load_size, agent.config.load_size)
        torch.onnx.export(local_model, sample_inputs.float().to(agent.config.device), os.path.join(agent.checkpoints_dir, "DQN.onnx"))
        # upload file to wandb
        wandb.save(os.path.join(agent.checkpoints_dir, "DQN.onnx"))

def setup_environment(config):
    envs = {}
    for vol_id in config.volume_ids.split(','):
        if not config.location_aware and not config.CT2US:
            envs[vol_id] = SingleVolumeEnvironment(config, vol_id=vol_id)
        elif not config.location_aware and config.CT2US: 
            envs[vol_id] = CT2USSingleVolumeEnvironment(config, vol_id=vol_id)
        elif config.location_aware and not config.CT2US:
            envs[vol_id] = LocationAwareSingleVolumeEnvironment(config, vol_id=vol_id)
        else:
            envs[vol_id] = LocationAwareCT2USSingleVolumeEnvironment(config, vol_id=vol_id)    
    # start reward function of each agent based on the input config parsed
    for key in envs:
        envs[key].set_reward()     
    return envs

def setup_optimizer(config, local_model):
    optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
    # load optimizer if needed
    if config.load is not None:
        if config.load_name is not None:
            load_name = config.load_name
        else:
            load_name = config.name
        print("loading {}/{} optimizer ...".format(load_name, config.load))
        state_dict = torch.load(os.path.join(config.checkpoints_dir, load_name, config.load+"_optimizer.pth"), map_location='cpu')
        optimizer.load_state_dict(state_dict)
    return optimizer

def setup_buffers(config):
    # instanciate buffers
    buffers = {}
    for vol_id in config.volume_ids.split(','):
        if config.recurrent:
            buffers[vol_id] = RecurrentPrioritizedReplayBuffer(config.buffer_size, config.batch_size, config.alpha, config.recurrent_history_len)
        else:
            buffers[vol_id] = PrioritizedReplayBuffer(config.buffer_size, config.batch_size, config.alpha)
    # load buffers if needed
    if config.load is not None:
        if config.load_name is not None:
            load_name = config.load_name
        else:
            load_name = config.name
        print("loading {}/{} buffers ...".format(load_name, config.load))
        for vol_id, buffer in buffers.items():
            buffer.load(os.path.join(config.checkpoints_dir, load_name, "{}_{}_".format(config.load, vol_id)))
    return buffers

def setup_criterion(config):
    if "mse" in config.loss.lower():
        criterion = nn.MSELoss(reduction='none')
    elif "smooth" in config.loss.lower():
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError()
    return criterion