from agent.agent import *
from utils import setup_environment
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options, load_options
from visualisation.visualizers import Visualizer
import torch, os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 1. gather options and load options from the training option file (or any other .txt file)
    parser = gather_options(phase="test")
    config = parser.parse_args()
    config.use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda" if config.use_cuda else "cpu")
    config = load_options(config, config.option_file)
    # set some defaults for testing:
    config.anatomyRewardWeight = 1
    config.planeDistanceRewardWeight = 1
    print_options(config, parser, save=False)
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
    rewards_anatomy = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    rewards_plane_distance = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    distance_from_goal = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    terminal_planes = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    goal_planes = {vol_id: env.sample_plane(env.goal_state, preprocess = True)["plane"].squeeze() for vol_id, env in envs.items()}
    for run in range(max(int(config.n_runs/len(envs)), 1)):
        print("test run: [{}]/[{}]".format(run+1, int(config.n_runs/len(envs))))
        out = agent.test_agent(config.n_steps, envs, qnetwork)
        for vol_id, logs in out.items():
            rewards_anatomy[vol_id].append(logs["logs_mean"]["anatomyReward"])
            rewards_plane_distance[vol_id].append(logs["logs_mean"]["planeDistanceReward"])
            distance_from_goal[vol_id].append(envs[vol_id].rewards["planeDistanceReward"].get_distance_from_goal(envs[vol_id].get_plane_coefs(*logs["states"][-1])))
            terminal_planes[vol_id].append(logs["planes"][-1])

    # extract mean and std for every vol_id, save terminal planes reached by the agent to disk
    for vol_id in config.volume_ids.split(","):
        rewards_anatomy[vol_id] = (np.mean(rewards_anatomy[vol_id]), np.std(rewards_anatomy[vol_id]))
        rewards_plane_distance[vol_id] = (np.mean(rewards_plane_distance[vol_id]), np.std(rewards_plane_distance[vol_id]))
        distance_from_goal[vol_id] = (np.mean(distance_from_goal[vol_id]), np.std(distance_from_goal[vol_id]))
        # save terminal planes to disk
        folder = "terminal_planes"
        if config.easy_objective: folder += "/easy_objective"
        else: folder += "/full_objective"
        if not os.path.exists(os.path.join(agent.results_dir, "test", folder, vol_id)):
            os.makedirs(os.path.join(agent.results_dir, "test", folder, vol_id))
        for i, sample in enumerate(terminal_planes[vol_id]):
            plt.imsave(os.path.join(agent.results_dir, "test", folder, vol_id, "sample%d.png"%i), sample, cmap="Greys_r")
        # save goal plane to disk
        if not os.path.exists(os.path.join(agent.results_dir, "test", "goal_planes", vol_id)):
            os.makedirs(os.path.join(agent.results_dir, "test", "goal_planes", vol_id))
        plt.imsave(os.path.join(agent.results_dir, "test", "goal_planes", vol_id, "goal_plane.png"), goal_planes[vol_id], cmap="Greys_r")        

    # print quantitative logs to .txt file
    message = "\t\t\t\t===== AVERAGE QUANTITATIVE RESULTS FOR : {} ===== \n\n".format(config.name)
    for vol_id in config.volume_ids.split(","):
        message += "{:10}anatomy reward: {:.4f} +/- {:.4f}\tplane distance reward: {:.4f} +/- {:.4f}\tdistance from goal: {:.4f} +/- {:.4f}\n".format(vol_id, 
                                                                                                                                                      rewards_anatomy[vol_id][0],
                                                                                                                                                      rewards_anatomy[vol_id][1],
                                                                                                                                                      rewards_plane_distance[vol_id][0],
                                                                                                                                                      rewards_plane_distance[vol_id][1],
                                                                                                                                                      distance_from_goal[vol_id][0],
                                                                                                                                                      distance_from_goal[vol_id][1])  
    
    name = config.name
    if config.easy_objective: name += "easy_objective"
    else: name += "full_objective"
    with open(os.path.join(agent.results_dir, "test", "{}.txt".format(name)), "wt") as out_file:
        out_file.write(message)
        out_file.write('\n')