from agent.agent import Agent
from environment.xcatEnvironment import SingleVolumeEnvironment
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
from visualisation.visualizers import Visualizer
import torch, os
import numpy as np

if __name__ == "__main__":
    # 1. gather options
    parser = gather_options(phase="test")
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
    # 4. instanciate Qnetwork
    qnetwork, _ = setup_networks(config)
    # 5. create results    
    visualizer  = Visualizer()
    out = agent.test_agent(config.n_steps, env, qnetwork)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))
    visualizer.render_full(out, fname = os.path.join(agent.results_dir, "test", "{}.gif".format(config.fname)))