"""Test script for multi-agent navigation towards a standard 4-Chamber view, can launch as:
>>>
nohup python test.py --name [experiment_name] --volume_ids samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7,samp8,samp9,samp10,samp11,samp12,samp13,samp14,
samp15,samp16,samp17,samp18,samp19 --n_runs 2000 --load latest --fname quantitative_metrics > out_filename.out

it will load training options from [--checkpoints_dir]/[experiment_name]/train_options.txt
see options/option.py for more info on possible options
"""

from agent.agent import *
from environment.xcatEnvironment import *
from environment.CT2USenvironment import *
from environment.realCTenvironment import *
from networks.Qnetworks import setup_networks
from options.options import gather_options, print_options, load_options
import torch, os
from tqdm import tqdm
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
    # 3. instanciate agent
    agent = SingleVolumeAgent(config)
    # 4. instanciate Qnetwork and set it in eval mode 
    qnetwork, _ = setup_networks(config)
    if not os.path.exists(os.path.join(agent.results_dir, "test")):
        os.makedirs(os.path.join(agent.results_dir, "test"))

    # 6. run test experiments on all given environments and generate outputs
    rewards_anatomy = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    rewards_plane_distance = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    distance_from_goal = {vol_id: [] for vol_id in config.volume_ids.split(",")}
    n_runs = max(int(config.n_runs/len(config.volume_ids.split(","))), 1)
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
        # for each test run
        for run in tqdm(range(n_runs), 'testing vol: {} ...'.format(vol_id)):
            print("test run: [{}]/[{}] env_subset [{}/[{}]".format(run+1, n_runs, i+1, len(config.volume_ids.split(","))))
            # pass in the corresponding subset of envs to test
            logs = agent.test_agent(config.n_steps, env, qnetwork)
            # store quantitative metrics
            if not config.realCT:
                rewards_anatomy[vol_id].append(logs["logs_mean"]["anatomyReward"])
                rewards_plane_distance[vol_id].append(logs["logs_mean"]["planeDistanceReward"])
                distance_from_goal[vol_id].append(env.rewards["planeDistanceReward"].get_distance_from_goal(env.get_plane_coefs(*logs["states"][-1])))
            # save terminal plane reached
            if config.location_aware:
                terminal_plane = logs["planes"][-1][0, ...].squeeze()
            else:
                terminal_plane = logs["planes"][-1]
            if not os.path.exists(os.path.join(agent.results_dir, "test", "terminal_planes", vol_id)):
                os.makedirs(os.path.join(agent.results_dir, "test", "terminal_planes", vol_id))
            plt.imsave(os.path.join(agent.results_dir, "test", "terminal_planes", vol_id, "sample{}.png".format(run)), terminal_plane, cmap="Greys_r")

    if not config.realCT:
        # extract mean and std for every vol_id, save terminal planes reached by the agent to disk
        for vol_id in config.volume_ids.split(","):
            rewards_anatomy[vol_id] = (np.mean(rewards_anatomy[vol_id]), np.std(rewards_anatomy[vol_id]))
            rewards_plane_distance[vol_id] = (np.mean(rewards_plane_distance[vol_id]), np.std(rewards_plane_distance[vol_id]))
            distance_from_goal[vol_id] = (np.mean(distance_from_goal[vol_id]), np.std(distance_from_goal[vol_id]))   
                
        # print quantitative logs to .txt file
        message = "\t\t\t\t===== AVERAGE QUANTITATIVE RESULTS FOR : {} ===== \n\n".format(config.name)
        for vol_id in config.volume_ids.split(","):
            message += "{:10}plane distance reward: {:.4f} +/- {:.4f}\tanatomy reward: {:.4f} +/- {:.4f}\tdistance from goal: {:.8f} +/- {:.8f}\n".format(vol_id, 
                                                                                                                                                        rewards_plane_distance[vol_id][0],
                                                                                                                                                        rewards_plane_distance[vol_id][1],
                                                                                                                                                        rewards_anatomy[vol_id][0],
                                                                                                                                                        rewards_anatomy[vol_id][1],
                                                                                                                                                        distance_from_goal[vol_id][0],
                                                                                                                                                        distance_from_goal[vol_id][1])  
        # get relative difference in mean performance between train and testing volumes
        plane_mean_train = np.mean([rewards_plane_distance[vol_id][0] for vol_id in config.volume_ids.split(",")[:15]])  
        anatomy_mean_train = np.mean([rewards_anatomy[vol_id][0] for vol_id in config.volume_ids.split(",")[:15]])
        distance_mean_train = np.mean([distance_from_goal[vol_id][0] for vol_id in config.volume_ids.split(",")[:15]])

        plane_mean_test = np.mean([rewards_plane_distance[vol_id][0] for vol_id in config.volume_ids.split(",")[15:]])  
        anatomy_mean_test = np.mean([rewards_anatomy[vol_id][0] for vol_id in config.volume_ids.split(",")[15:]])
        distance_mean_test = np.mean([distance_from_goal[vol_id][0] for vol_id in config.volume_ids.split(",")[15:]])
                                                                                                                                            
        plane_mean = plane_mean_test/plane_mean_train*100
        anatomy_mean = anatomy_mean_test/anatomy_mean_train*100
        distance_mean = distance_mean_test/distance_mean_train*100

        # print mean performances
        message += "\n\nmean performances: plane distance reward: {:.4f}\tanatomy reward: {:.4f}\tdistance from goal: {:.8f}\n".format(plane_mean,
                                                                                                                                    anatomy_mean,
                                                                                                                                    distance_mean)
        # print output in latex code 
        message += "\t\t\t\t===== LATEX TABLE CODE FOR : {} ===== \n\n".format(config.name)
        for vol_id in config.volume_ids.split(","):
            message += "{:10}->\t${:.4f} \pm {:.4f}$ & ${:.4f} \pm {:.4f}$ & ${:.8f} \pm {:.8f}$\\\n".format(vol_id, 
                                                                                                            rewards_plane_distance[vol_id][0],
                                                                                                            rewards_plane_distance[vol_id][1],
                                                                                                            rewards_anatomy[vol_id][0],
                                                                                                            rewards_anatomy[vol_id][1],
                                                                                                            distance_from_goal[vol_id][0],
                                                                                                            distance_from_goal[vol_id][1]) 
        message += "\n\nmean performances: \t${:.2f}$ & ${:.2f}$ & ${:.2f}$\\\n".format(plane_mean,
                                                                                        anatomy_mean,
                                                                                        distance_mean)    

        name = config.name
        if config.easy_objective: name += "easy_objective"
        else: name += "full_objective"
        with open(os.path.join(agent.results_dir, "test", "{}.txt".format(name)), "wt") as out_file:
            out_file.write(message)
            out_file.write('\n')