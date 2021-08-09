from agent.agent import *
from utils import setup_environment
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
from visualisation.visualizers import Visualizer
import torch, os, json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
    # 5. instanciate visualizer to plot results    
    visualizer  = Visualizer(agent.results_dir)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))
    # 6. run test experiments on all given environments and generate outputs
    total_rewards, last_frames = {},{}
    if config.mainReward in ["both", "planeDistanceReward"]:
        total_rewards["planeDistanceReward"] = {}
    if config.mainReward in ["both", "anatomyReward"]:
        total_rewards["anatomyReward"] = {}
    for run in range(max(int(config.n_runs/len(envs)), 1)):
        print("test run: [{}]/[{}]".format(run+1, int(config.n_runs/len(envs))))
        out = agent.test_agent(config.n_steps, envs, qnetwork)
        for i, (key, logs) in enumerate(out.items(), 1):
            # 6.1. gather total rewards accumulated in testing episode
            for reward in total_rewards:
                if key not in total_rewards[reward]:
                    total_rewards[reward][key] = []
                total_rewards[reward][key].append(logs["logs"][reward])
            # 6.2. store last frame for visualization later
            if len(logs["planes"][-1]>2):
                last_frames[key] = logs["planes"][-1][0, ...].squeeze()
            else:
                last_frames[key] = logs["planes"][-1]
            last_frames[key]/=last_frames[key].max()
            # 6.3. render trajectories if queried
            if config.render:
                print("rendering logs for: {} ([{}]/[{}])".format(key, i, len(out)))
                if not os.path.exists(os.path.join(agent.results_dir, "test", key)):
                    os.makedirs(os.path.join(agent.results_dir, "test", key))
                visualizer.render_full(logs, fname = os.path.join(agent.results_dir, "test", key, "{}_{}.gif".format(config.fname, run)))
                #visualizer.render_frames(logs["planes"], logs["planes"], fname="trajectory.gif", n_rows=2, fps=10)
    
    # 7. re-organize logged rewards
    for reward_key, reward in total_rewards.items():
        fig = plt.figure()
        ax = plt.gca()
        for vol_id, log in reward.items():
            log = np.array(log).astype(np.float)
            means = log.mean(0)
            stds = log.std(0)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(range(len(means)), means, label=vol_id, c=color)
            plt.fill_between(range(len(means)), means-stds, means+stds ,alpha=0.3, facecolor=color)
            ax.figure.figimage(last_frames[vol_id],len(means)-1,means[-1], alpha=0.5)
            imagebox = OffsetImage(last_frames[vol_id], zoom=0.2)
            ab = AnnotationBbox(imagebox, ((len(means)-1)/ax.get_xlim()[-1], means[-1]/ax.get_ylim()[-1]))
        plt.legend()
        plt.title("average {} collected in an episode".format(reward_key))
        # 8. save figure
        if not os.path.exists(os.path.join(agent.results_dir, "test")):
            os.makedirs(os.path.join(agent.results_dir, "test"))
        plt.savefig(os.path.join(agent.results_dir, "test", "{}_test.pdf".format(reward_key)))
