from environment.xcatEnvironment import SingleVolumeEnvironment
from agent.agent import Agent
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
from tqdm import tqdm
import torch, os, wandb
import torch.multiprocessing as mp

def train_agent(agent, env, local_model, target_model, optimizer, criterion, rank=0):
        """ Trains an agent on an input environment, given networks/optimizers and training criterions.
        Params:
        ==========
                agent (agent/* instance): the agent class.
                env (environment/* instance): the environment the agent will interact with while training.
                local_model (PyTorch model): pytorch network that will be trained using a particular training routine (i.e. DQN)
                target_model (PyTorch model): pytorch network that will be used as a target to estimate future Qvalues. 
                                                (it is a hard copy or a running average of the local model, helps against diverging)
                optimizer (PyTorch optimizer): optimizer to update the local network weights.
                criterion (PyTorch Module): loss to minimize in order to train the local network.
                rank (int): indicates the process number if multiple processes are queried
        """ 
        # manual seed
        torch.manual_seed(agent.config.seed + rank) 
        # 1. launch exploring steps if needed
        if agent.config.exploring_steps>0:
                print("random walk to collect experience...")
                env.random_walk(agent.config.exploring_steps, agent.config.exploring_restarts)  
        # 2. initialize wandb for logging purposes
        if agent.config.wandb in ["online", "offline"]:
                wandb.login()
        wandb.init(project="AutomaticUSnavigation", name=agent.config.name, group=agent.config.name, config=agent.config, mode=agent.config.wandb)
        # 3. tell wandb to watch what the model gets up to: gradients, weights, and loss
        wandb.watch(local_model, criterion, log="all", log_freq=agent.config.log_freq)
        # 4. start training
        for episode in tqdm(range(agent.config.n_episodes), desc="training..."):
                logs = agent.play_episode(env, local_model, target_model, optimizer, criterion)
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

        # 1. gather options
        parser = gather_options(phase="train")
        config = parser.parse_args()
        config.use_cuda = torch.cuda.is_available()
        config.device = torch.device("cuda" if config.use_cuda else "cpu")
        print_options(config, parser)

        # 2. instanciate environment(s)
        vol_ids = config.volume_ids.split(",")
        if len(vol_ids)>1:
                raise ValueError('only supporting single volume environments for now.')
                # envs = []
                # for vol_id in range(len(vol_ids)):
                #         envs.append(SingleVolumeEnvironment(config, vol_id=vol_id))
        else:
                env = SingleVolumeEnvironment(config)
        # 3. instanciate agent
        agent = Agent(config)
        # 4. instanciate Qnetworks, optimizer and training criterion
        qnetwork_local, qnetwork_target, optimizer, criterion = setup_networks(config)
        qnetwork_local.share_memory() # gradients are allocated lazily, so they are not shared here, necessary to train on multiple processes
        # 5. launch training
        # MULTI-PROCESS TRAINING
        torch.manual_seed(config.seed)
        if config.n_processes>1:
                mp.set_start_method('spawn')
                processes = []
                for rank in range(config.n_processes):
                        p = mp.Process(target=train_agent, args=(agent, env, qnetwork_local, qnetwork_target, optimizer, criterion, rank))
                        p.start()
                        processes.append(p)
                for p in processes:
                        p.join()
        # SINGLE PROCESS TRAINING
        else:
                train_agent(agent, env, qnetwork_local, qnetwork_target, optimizer, criterion)

