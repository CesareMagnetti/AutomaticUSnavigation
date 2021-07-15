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

def test(config):
    """ tests a trained agent.
    Params:
    ==========
            config (argparse object): configuration with all training options. (see options/options.py)
    """ 
    # ==== instanciate useful classes ====

    # manual seed
    torch.manual_seed(config.seed) 
    # 1. instanciate environment
    env = SingleVolumeEnvironment(config)
    # 2. instanciate agent
    agent = Agent(config)
    # 3. instanciate optimizer for local_network
    optimizer = optim.Adam(local_model.parameters(), lr=config.learning_rate)
    # 4. instanciate Qnetwork
    qnetwork, _ = setup_networks(config) 
    # 5. initialize wandb for logging purposes
    if config.wandb in ["online", "offline"]:
            wandb.login()
    ## uncomment the next lines when not performing a sweep.
    config.name = "{}_anatomyReward_{}_areaReward_{}_steppingReward_{}_oobReward_{}_stopReward".format(config.anatomyRewardIDs,
                                                                                                       config.areaRewardWeight,
                                                                                                       config.steppingReward,
                                                                                                       config.oobReward,
                                                                                                       config.stopReward)
    wandb.init(entity="us_navigation", config=config, mode=config.wandb, name=config.name)
    config = wandb.config # oddly this ensures wandb works smoothly
    # 4. start testing
    anatomy_reward_collected = []
    for run in tqdm(range(config.n_runs), desc="testing..."):
            out = agent.test_agent(config.n_steps, env, qnetwork)
            anatomy_reward_collected.append(out["wandb"]["anatomyReward_test"])
    # 5. send relevant testing log to wandb
    wandb.log({"testingAnatomyReward": np.mean(anatomy_reward_collected), commit=True)

if __name__=="__main__":
    # 1. gather options
    parser = gather_options(phase="test")
    config = parser.parse_args()
    config.use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda" if config.use_cuda else "cpu")
    print_options(config, parser)
    # 2. launch testing
    test(config)