from agent.agent import *
from utils import setup_environment
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
    envs = setup_environment(config)
    # 3. instanciate agent
    agent = MultiVolumeAgent(config)
    # 4. instanciate Qnetwork and set it in eval mode 
    qnetwork, _ = setup_networks(config)
    qnetwork.eval()
    # 5. instanciate visualizer to plot results    
    visualizer  = Visualizer(agent.results_dir)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))
    # 6. run test experiments on all given environments and generate outputs
    for run in range(max(int(config.n_runs/len(envs)), 1)):
        print("test run: [{}]/[{}]".format(run+1, int(config.n_runs/len(envs))))
        if not config.CT2US:
            out = agent.test_agent(config.n_steps, envs, qnetwork)
            for i, (key, logs) in enumerate(out.items(), 1):
                print("rendering logs for: {} ([{}]/[{}])".format(key, i, len(out)))
                if not os.path.exists(os.path.join(agent.results_dir, "test", key)):
                    os.makedirs(os.path.join(agent.results_dir, "test", key))
                visualizer.render_full(logs, fname = os.path.join(agent.results_dir, "test", key, "{}_{}.gif".format(config.fname, run)))
        else:
            out = agent.test_agentUS(config.n_steps, envs, qnetwork)
            visualizer.render_frames_double(out["framesCT"], out["frames"], fname="US_trajectory.gif")