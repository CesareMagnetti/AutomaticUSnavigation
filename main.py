from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
from logger.logger import Log, Logger
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
from collections import deque
import numpy as np

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
parser.add_argument('--train', action='store_true', help='if training the agent before testing it.')
parser.add_argument('--savedir', '-s', type=str, default='./results/', help='where to save the trajectory.')
parser.add_argument('--name', '-n', type=str, default='sample_experiment', help='name of the experiment.')
parser.add_argument('--volume_id', '-vol_id', type=str, default='default_512_CT_1', help='filename of the CT volume.')
parser.add_argument('--segmentation_id', '-seg_id', type=str, default='default_512_SEG_1', help='filename of the segmented CT volume.')
parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1', help='filename for the state dict of the ct2us model (.pth) file.\n'\
                                                                      'available models can be found at ./models')
parser.add_argument('--batch_size', type=int, default=32, help="batch size for the replay buffer.")
parser.add_argument('--buffer_size', type=int, default=int(1e5), help="capacity of the replay buffer.")
parser.add_argument('--gamma', type=int, default=0.99, help="discount factor.")
parser.add_argument('--tau', type=int, default=1e-3, help="weight for soft update of target parameters.")
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help="learning rate for the q network.")
parser.add_argument('--update_every', type=int, default=4, help="how often to update the network, in steps.")



args = parser.parse_args()

def train(n_episodes=2000, n_steps_per_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    rewards, TDerrors = Log("rewards", n_episodes, n_steps_per_episode), Log("TDerrors", n_episodes, n_steps_per_episode)
    logger = Logger(os.path.join(args.savedir, args.name), rewards, TDerrors)
    # initialize epsilon
    eps = eps_start 
    print_every = 20
    for episode in range(1, n_episodes+1):
        env.reset()
        state, _, _ = env.sample()
        for _ in range(n_steps_per_episode):
            # get action from current state
            actions = agent.act(state, eps)
            # observe next state and reward 
            next_state, reward, _ = env.step(*actions) 
            # update Q network using Q learning algo, return the TD error
            TDerror = agent.step(state.detach().cpu(),
                                 actions,
                                 reward,
                                 next_state.detach().cpu())
            state = next_state
            # add logs
            rewards.push(reward)
            if TDerror: # TD error can be None if agent.update_every>1
                TDerrors.push(TDerror)

        if episode % print_every == 0:
            print("[{}/{}] ({:.0f}%) eps:{:.3f}\n\t\ttotal reward collected in the last episode: {:.2f}\n\t\t"\
                  "mean reward collected in previous episodes: {:.2f}".format(episode,
                                                                              n_episodes,
                                                                              int(episode/n_episodes*100),
                                                                              eps,
                                                                              np.sum(rewards.current(), axis=-1),
                                                                              np.sum(rewards.mean(episodes=slice(0,episode)), axis=-1)))
            logger.current_visuals(save=True, with_total_reward=True)
            logger.save_current_logs_to_txt()
        
        # update eps
        eps = max(eps*eps_decay, eps_end)
        # step logs for next episode
        logger.step()


def test(max_t=250):
    env.reset()
    state, _, _ = env.sample()
    frames=[]
    for i in tqdm(range(max_t)):
        actions = agent.act(state)
        state, reward, _ = env.step(*actions)
        frames.append(env.render(state, titleText='time step: {} reward:{:.5f}'.format(i+1, reward)))
  
    # save all frames as a GIF
    if not os.path.exists(os.path.join(args.savedir, args.name)):
        os.makedirs(os.path.join(args.savedir, args.name))

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name, 'navigation.gif'), fps=10)



if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args.dataroot, args.volume_id, args.segmentation_id, use_cuda=True)
    state, _, _ = env.sample()

    # instanciate agent (3 agents: 1 for each point. 6 actions: up/down, left/right, forward/backwards)
    agent = Agent(state_size=state.shape, action_size=6, Nagents=3, seed=1, use_cuda=True)

    # train agent
    if args.train:
        train()
    
    # test agent
    test()
