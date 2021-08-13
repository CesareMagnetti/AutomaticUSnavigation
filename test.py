from agent.agent import *
from utils import setup_environment
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options
from visualisation.visualizers import Visualizer
import torch, os, json
import numpy as np
# visualization
from matplotlib import pyplot as plt
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage

# handler class to insert images in the legend (gives insights of size of heach heart volume tested)
class HandlerLineImage(HandlerBase):

    def __init__(self, img_arr, space=15, offset = 10 ):
        self.space=space
        self.offset=offset
        self.image_data = img_arr        
        super(HandlerLineImage, self).__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        l = matplotlib.lines.Line2D([xdescent+self.offset,xdescent+(width-self.space)/3.+self.offset],
                                     [ydescent+height/2., ydescent+height/2.])
        l.update_from(orig_handle)
        l.set_clip_on(False)
        l.set_transform(trans)

        bb = Bbox.from_bounds(xdescent +(width+self.space)/3.+self.offset,
                              ydescent,
                              height*self.image_data.shape[1]/self.image_data.shape[0],
                              height)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        image.set_cmap("Greys_r")

        self.update_prop(image, orig_handle, legend)
        return [l,image]

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
    total_rewards = {}
    # get the goal plane for each env for comparison in the visualization
    goal_planes = {env.vol_id: env.sample_plane(env.goal_state)["plane"] for env in envs}
    for key,value in goal_planes.items():
        if len(value.shape)>2:
            goal_planes[key] =  value[0, ...]
        goal_planes[key] = goal_planes[key]/goal_planes[key].max()
    total_rewards["planeDistanceReward"] = {}
    if config.anatomyRewardWeight > 0:
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
            # 6.2. render trajectories if queried
            if config.render:
                print("rendering logs for: {} ([{}]/[{}])".format(key, i, len(out)))
                if not os.path.exists(os.path.join(agent.results_dir, "test", key)):
                    os.makedirs(os.path.join(agent.results_dir, "test", key))
                visualizer.render_full(logs, fname = os.path.join(agent.results_dir, "test", key, "{}_{}.gif".format(config.fname, run)))
                #visualizer.render_frames(logs["planes"], logs["planes"], fname="trajectory.gif", n_rows=2, fps=10)
    
    # 7. re-organize logged rewards
    for reward_key, reward in total_rewards.items():
        fig = plt.figure(figsize=(15,10))
        ax = plt.gca()
        lines = {}
        last_reward = []
        for vol_id, log in reward.items():
            log = np.array(log).astype(np.float)
            means = log[:,1:].mean(0)
            stds = log[:,1:].std(0)
            last_reward.append(means[-1])
            color = next(ax._get_lines.prop_cycler)['color']
            lines[vol_id], = plt.plot(range(len(means)), means, c=color)
            plt.fill_between(range(len(means)), means-stds, means+stds ,alpha=0.3, facecolor=color)

        # add legend with a reference image to compare heart sizes
        line_values, line_keys, imgs, last_reward = zip(*sorted(zip(lines.values(), lines.keys(), goal_planes.values(), last_reward), key=lambda t: t[-1], reverse=True))
        # add legend with vol_ids
        legend1 = plt.legend(line_values, line_keys, loc="lower center", ncol=len(line_keys))
        plt.legend(line_values,
                   [""]*len(lines),
                   handler_map={line: HandlerLineImage(img) for line,img in zip(line_values,imgs)}, 
                   handlelength=0.25, fontsize=80, labelspacing=0., bbox_to_anchor=(0.94,1.1), frameon=False)
        ax.add_artist(legend1)
        plt.title("average {} collected in an episode".format(reward_key))
        # 8. save figure
        if not os.path.exists(os.path.join(agent.results_dir, "test")):
            os.makedirs(os.path.join(agent.results_dir, "test"))
        plt.savefig(os.path.join(agent.results_dir, "test", "{}_test.pdf".format(reward_key)))
