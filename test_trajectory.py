"""Test trajectory script for multi-agent navigation towards a standard 4-Chamber view, can launch as:
>>>
python test_trajectory.py --name [experiment_name] --volume_ids samp15,samp16,samp17,samp18,samp19 --n_steps 250 --load latest

it will load training options from [--checkpoints_dir]/[experiment_name]/train_options.txt
see options/option.py for more info on possible options
"""

from agent.agent import *
from environment.xcatEnvironment import *
from environment.CT2USenvironment import *
from environment.realCTenvironment import *
from visualisation.visualizers import Visualizer
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options, load_options

if __name__ == "__main__":
    # 1. gather options and load options from the training option file (or any other .txt file)
    parser = gather_options(phase="test")
    config = parser.parse_args()
    config.use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda" if config.use_cuda else "cpu")
    if not config.no_load:
        config = load_options(config, config.option_file)
    # set some defaults for testing:
    config.anatomyRewardWeight = 1
    config.planeDistanceRewardWeight = 1
    print_options(config, parser, save=False)
    # 3. instanciate agent
    agent = SingleVolumeAgent(config)
    # 4. instanciate Qnetwork and set it in eval mode 
    qnetwork, _ = setup_networks(config)
    # 6. instanciate visualizer
    visualizer = Visualizer(agent.results_dir)
    # test trajectory on each environment
    for i, vol_id in enumerate(config.volume_ids.split(",")):
        # setup environment
        if config.realCT:
            env = realCTtestEnvironment(config, vol_id=vol_id)
        elif not config.location_aware and not config.CT2US:
            env = SingleVolumeEnvironment(config, vol_id=vol_id)
        elif not config.location_aware and config.CT2US: 
            env = CT2USSingleVolumeEnvironment(config, vol_id=vol_id)
        elif config.location_aware and not config.CT2US:
            env = LocationAwareSingleVolumeEnvironment(config, vol_id=vol_id)
        else:
            env = LocationAwareCT2USSingleVolumeEnvironment(config, vol_id=vol_id)  
        if not config.realCT:
            env.set_reward() # this starts the reward logs with the config file parsed before

        # pass in the corresponding subset of envs to test
        logs = agent.test_agent(config.n_steps, env, qnetwork)
        # animate the trajectory followed by the agent in the current episode
        if agent.config.CT2US:
            visualizer.render_frames(logs["planes"], logs["planesCT"], n_rows = 2 if agent.config.location_aware else 1, fname = "{}_{}.gif".format(config.fname, vol_id))
        else:
            visualizer.render_frames(logs["planes"], n_rows = 2 if agent.config.location_aware else 1, fname = "{}_{}.gif".format(config.fname, vol_id))