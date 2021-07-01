from agent.agent import Agent
from environment.xcatEnvironment import SingleVolumeEnvironment
from networks.Qnetworks import SimpleQNetwork
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
    qnetwork = SimpleQNetwork((1, config.load_size, config.load_size), config.action_size, config.n_agents, config.seed, config.n_blocks_Q,
                                config.downsampling_Q, config.n_features_Q, config.dropout_Q).to(config.device)
    if config.load is not None:
        print("loading: {} ...".format(config.load))
        qnetwork.load(os.path.join(config.checkpoints_dir, config.name, config.load+".pth"))
    
    visualizer  = Visualizer()
    out = agent.test_agent(config.n_steps, env, qnetwork)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))
    visualizer.render_full(out, fname = os.path.join(agent.results_dir, "test", "sample.gif"))