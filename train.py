from environment.xcatEnvironment import SingleVolumeEnvironment
from agent.agent import Agent
from networks.Qnetworks import setup_networks
from buffer.buffer import *
from options.options import gather_options, print_options
from visualisation.visualizers import Visualizer
from tqdm import tqdm
import torch, os, wandb
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

def train(config, local_model, target_model, rank=0):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                config (argparse object): configuration with all training options. (see options/options.py)
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                                             (if more processes, the Qnetwork is shared)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                              (it is a hard copy or a running average of the local model, helps against diverging)
                rank (int): indicates the process number if multiple processes are queried
        """ 
        # ==== instanciate useful classes ====

        # manual seed
        torch.manual_seed(config.seed + rank) 
        # 1. instanciate environment
        env = SingleVolumeEnvironment(config)
        # 2. instanciate agent
        agent = Agent(config)
        # 3. instanciate optimizer for local_network
        optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
        # 4. instanciate criterion
        if "mse" in config.loss.lower():
                criterion = nn.MSELoss(reduction='none')
        elif "smooth" in config.loss.lower():
                criterion = nn.SmoothL1Loss(reduction='none')
        else:
                raise ValueError()
        # 5. instanciate replay buffer
        if config.alpha>0:
                buffer = PrioritizedReplayBuffer(config.buffer_size, config.batch_size, config.alpha)
        else:
                buffer = ReplayBuffer(config.buffer_size, config.batch_size)
        # 6. instanciate visualizer
        visualizer = Visualizer()

        # ==== LAUNCH TRAINING ====

        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                print("random walk to collect experience...")
                env.random_walk(config.exploring_steps, buffer, config.exploring_restarts)  
        # 2. initialize wandb for logging purposes
        if config.wandb in ["online", "offline"]:
                wandb.login()
        ## uncomment this when not performing a sweep and comment the next line.
        wandb.init(entity="us_navigation", config=config, mode=config.wandb)#name=config.name)
        #wandb.init(config=config, entity="cesare-magnetti", project="AutomaticUSnavigation")
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
                        print("saving latest model weights...")
                        local_model.save(os.path.join(agent.checkpoints_dir, "latest.pth"))
                        target_model.save(os.path.join(agent.checkpoints_dir, "episode%d.pth"%episode))
                        # test the greedy policy and send logs
                        out = agent.test_agent(config.n_steps_per_episode, env, local_model)
                        wandb.log(out["wandb"], commit=True)
                        # plot the trajectory followed by the agent in the current episode
                        if not os.path.exists(os.path.join(agent.results_dir, "visuals")):
                                os.makedirs(os.path.join(agent.results_dir, "visuals"))
                        fname = os.path.join(agent.results_dir, "visuals", "episode%d.gif"%episode)
                        visualizer.render_frames(out["frames"], fname)
                        wandb.save(fname)

if __name__=="__main__":

        # 2. gather options
        parser = gather_options(phase="train")
        config = parser.parse_args()
        config.use_cuda = torch.cuda.is_available()
        config.device = torch.device("cuda" if config.use_cuda else "cpu")
        print_options(config, parser)
        
        # 2. instanciate Qnetworks
        qnetwork_local, qnetwork_target = setup_networks(config)
        qnetwork_local.share_memory()  # gradients are allocated lazily, so they are not shared here, necessary to train on multiple processes
        # 3. launch training
        # MULTI-PROCESS TRAINING
        if config.n_processes>1:
                raise NotImplementedError("MULTI-PROCESSING DOES NOT CURRENTLY WORK.")
                mp.set_start_method('spawn')
                processes = []
                for rank in range(config.n_processes):
                        p = mp.Process(target=train, args=(config, qnetwork_local, qnetwork_target, rank))
                        p.start()
                        processes.append(p)
                for p in processes:
                        p.join()
        # SINGLE PROCESS TRAINING
        else:
                train(config, qnetwork_local, qnetwork_target)

