from baseEnvironment import BaseEnvironment
import argparse
import numpy as np
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
import os

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
parser.add_argument('--root', '-r',  type=str, help='path to the CT volume.')
parser.add_argument('--Nsteps', '-N',  type=int, default=100, help='how many steps to take.')
parser.add_argument('--savedir', '-s', type=str, default='./trajectories/', help='where to save the trajectory.')
parser.add_argument('--name', '-n', type=str, default='sample', help='name of the experiment.')

args = parser.parse_args()

if __name__=="__main__":
    env = BaseEnvironment(args.root)

    # get the current plane
    state = env.sample()

    frames = [state]
    for iter in range(args.Nsteps):

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
        state = env.step(*increments)
        frames.append(state)
    
    # save all frames as a GIF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name), fps=10)