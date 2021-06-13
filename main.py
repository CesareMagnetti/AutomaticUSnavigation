from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os
from collections import deque
import numpy as np

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
parser.add_argument('--volume_id', '-vol_id', type=str, default='default_512_CT_1', help='filename of the CT volume.')
parser.add_argument('--segmentation_id', '-seg_id', type=str, default='default_512_SEG_1', help='filename of the segmented CT volume.')
parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1', help='filename for the state dict of the ct2us model (.pth) file.\n'\
                                                                      'available models can be found at ./models')
parser.add_argument('--savedir', '-s', type=str, default='./results/', help='where to save the trajectory.')
parser.add_argument('--name', '-n', type=str, default='sample_experiment', help='name of the experiment.')

args = parser.parse_args()

def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for episode in range(1, n_episodes+1):
        env.reset()
        state, _, _ = env.sample()
        score = 0
        for t in range(max_t):
            actions = agent.act(state, eps)
            next_state, reward, _ = env.step(*actions)
            agent.step(state.detach().cpu(),
                       actions,
                       reward,
                       next_state.detach().cpu())
            state = next_state
            score += reward

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        #     agent.qnetwork_local.save(os.path.join(args.savedir, 'checkpoint.pth'))
        #     break
    return scores

def test(max_t=1000):
    frames=[]
    for i in tqdm(range(max_t)):
        actions = agent.act(state)
        frames.append(env.render(state, titleText='time step: %d'%(i+1)))
        state, reward, _ = env.step(*actions)
    
    # save all frames as a GIF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name+'.gif'), fps=10)



if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args.dataroot, args.volume_id, args.segmentation_id, use_cuda=True)
    state, _, _ = env.sample()

    # instanciate agent (3 agents: 1 for each point. 6 actions: up/down, left/right, forward/backwards)
    agent = Agent(state_size=state.shape, action_size=6, Nagents=3, seed=1, use_cuda=True)

    # train agent
    train()
    
    # test agent
    test()
