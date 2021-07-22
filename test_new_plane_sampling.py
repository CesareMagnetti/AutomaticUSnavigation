from agent.agent import Agent
from environment.xcatEnvironment import TestNewSamplingEnvironment
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
    # 2. instanciate environmentv(only single env for now)
    env = TestNewSamplingEnvironment(config)
    # 3. instanciate agent
    agent = Agent(config)
    # 4. instanciate Qnetwork
    qnetwork, _ = setup_networks(config)
    # 5. instanciate visualizer to plot results    
    visualizer  = Visualizer(agent.results_dir)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))
    # 6. run test experiments and generate outputs
    for run in range(config.n_runs):
        print("test run: [{}]/[{}]".format(run+1, config.n_runs))
        out = agent.test_agent(config.n_steps, env, qnetwork)
        visualizer.render_full(out, fname = os.path.join(agent.results_dir, "test", "{}_{}.gif".format(config.fname, run)))