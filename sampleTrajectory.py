from environment.baseEnvironment import BaseEnvironment
from environment.ct2usEnvironment import CT2USEnvironment
import argparse
import numpy as np
from moviepy.editor import ImageSequenceClip
import os
import cv2
from tqdm import tqdm

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
parser.add_argument('--volume_id', '-vol_id', type=str, default='default_512_CT_1', help='filename of the CT volume.')
parser.add_argument('--segmentation_id', '-seg_id', type=str, default='default_512_SEG_1', help='filename of the segmented CT volume.')
parser.add_argument('--ct2us_model_name', '-model', type=str, default='CycleGAN_LPIPS_noIdtLoss_lambda_AB_1', help='filename for the state dict of the ct2us model (.pth) file.\n'\
                                                                      'available models can be found at ./models')
parser.add_argument('--Nsteps', '-N',  type=int, default=100, help='how many steps to take.')
parser.add_argument('--savedir', '-s', type=str, default='./trajectories/', help='where to save the trajectory.')
parser.add_argument('--name', '-n', type=str, default='sample', help='name of the experiment.')

args = parser.parse_args()

if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args.dataroot, args.volume_id, args.segmentation_id)
    #env = CT2USEnvironment(args.dataroot, args.volume_id, args.segmentation_id, model_name=args.ct2us_model_name, use_cuda=True)

    frames = []
    for iter in tqdm(range(args.Nsteps)):

        # sample a random increment of +- 1 pixel for each point
        increments = []
        for point in range(3):
            # random movement in a dimension of each point
            dimension = np.random.choice([0,1,2])
            # random direction to move in
            direction = np.random.choice([-1,1])
            # define the increment
            incr = np.zeros((3,))
            incr[dimension] = direction
            increments.append(incr)
        
        # step the environment accordingly to get the new plane
        state, reward, seg = env.step(*increments)

        # append frame to array
        frames.append(env.render(state, seg, "reward: {}".format(reward)))

    
    # save all frames as a GIF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name+'.gif'), fps=10)