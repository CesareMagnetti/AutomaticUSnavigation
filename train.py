from environment.xcatEnvironment import SingleVolumeEnvironment
from agent.agent import Agent
from networks.Qnetworks import setup_networks
from buffer.buffer import ReplayBuffer
from options.options import gather_options, print_options
from tqdm import tqdm
import torch, os, wandb
import torch.multiprocessing as mp

def train(config, local_model, target_model, buffer, rank=0):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                config (argparse object): configuration with all training options. (see options/options.py)
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                                             (if more processes, the Qnetwork is shared)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                              (it is a hard copy or a running average of the local model, helps against diverging)
                buffer (buffer/* object): replay buffer shared amongst processes (each process pushes to the same memory.)
                rank (int): indicates the process number if multiple processes are queried
        """ 
        # ==== instanciate useful classes ====

        # manual seed
        torch.manual_seed(agent.config.seed + rank) 
        # 1. instanciate environment
        env = SingleVolumeEnvironment(config)
        # 2. instanciate agent
        agent = Agent(config)
        # 3. instanciate optimizer for local_network
        optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
        # 4. instanciate criterion
        if "mse" in config.loss.lower():
                criterion = nn.MSELoss()
        elif "smooth" in config.loss.lower():
                criterion = nn.SmoothL1Loss()
        else:
                raise ValueError()

        # ==== LAUNCH TRAINING ====

        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                print("random walk to collect experience...")
                env.random_walk(agent.config.exploring_steps, buffer, agent.config.exploring_restarts)  
        # 2. initialize wandb for logging purposes
        if agent.config.wandb in ["online", "offline"]:
                wandb.login()
        wandb.init(project="AutomaticUSnavigation", name=agent.config.name, group=agent.config.name, config=agent.config, mode=agent.config.wandb)
        # 3. tell wandb to watch what the model gets up to: gradients, weights, and loss
        wandb.watch(local_model, criterion, log="all", log_freq=agent.config.log_freq)
        # 4. start training
        for episode in tqdm(range(agent.config.n_episodes), desc="training..."):
                logs = agent.play_episode(env, local_model, target_model, optimizer, criterion, buffer)
                # send logs to weights and biases
                if episode % agent.config.log_freq == 0:
                        wandb.log(logs, step=agent.t_step, commit=True)
                # save agent locally and test its current greedy policy
                if episode % agent.config.save_freq == 0:
                        print("saving latest model weights...")
                        local_model.save(os.path.join(agent.checkpoints_dir, "latest.pth"))
                        target_model.save(os.path.join(agent.checkpoints_dir, "episode%d.pth"%episode))
                        # test current greedy policy
                        agent.test_agent(agent.config.n_steps_per_episode, env, local_model, "episode%d"%episode) 



if __name__=="__main__":

        # 2. gather options
        parser = gather_options(phase="train")
        config = parser.parse_args()
        config.use_cuda = torch.cuda.is_available()
        config.device = torch.device("cuda" if config.use_cuda else "cpu")
        print_options(config, parser)
        
        # 2. instanciate Qnetworks
        qnetwork_local, qnetwork_target = setup_networks(config)
        qnetwork_local.share_memory() # gradients are allocated lazily, so they are not shared here, necessary to train on multiple processes
        # 3. instanciate replay buffer
        buffer = ReplayBuffer(config.buffer_size, config.batch_size)
        # 4. launch training
        # MULTI-PROCESS TRAINING
        torch.manual_seed(config.seed)
        if config.n_processes>1:
                mp.set_start_method('spawn')
                processes = []
                for rank in range(config.n_processes):
                        p = mp.Process(target=train, args=(config, qnetwork_local, qnetwork_target, buffer, rank))
                        p.start()
                        processes.append(p)
                for p in processes:
                        p.join()
        # SINGLE PROCESS TRAINING
        else:
                train_agent(config, qnetwork_local, qnetwork_target, buffer)

