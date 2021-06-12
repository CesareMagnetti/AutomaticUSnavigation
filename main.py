from environment.baseEnvironment import BaseEnvironment
from agent.agent import Agent
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import os

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

if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args.dataroot, args.volume_id, args.segmentation_id, use_cuda=True)
    state, _, _ = env.sample()

    # instanciate agent (3 agents: 1 for each point. 6 actions: up/down, left/right, forward/backwards)
    agent = Agent(state_size=state.shape, action_size=6, Nagents=3, seed=1, use_cuda=True)

    # watch an untrained agent
    frames=[]
    for i in tqdm(range(100)):
        actions = agent.act(state)
        frames.append(env.render(state, titleText='time step: %d'%(i+1)))
        state, reward, _ = env.step(*actions)
    
    # save all frames as a GIF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name+'.gif'), fps=10)