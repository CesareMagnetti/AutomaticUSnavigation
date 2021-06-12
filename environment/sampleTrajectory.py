from baseEnvironment import BaseEnvironment
import argparse
import numpy as np
from moviepy.editor import ImageSequenceClip
import os
import cv2

def render_frame(state, seg=None, titleText=None):

    # stack image and segmentation, progate through channel dim since black and white
    # must do this to call ``ImageSequenceClip`` later.
    if seg is not None:
        image = np.hstack([state[..., np.newaxis] * np.ones(3), seg[..., np.newaxis] * np.ones(3)])
    else:
        image = state.copy()

    # put title on image
    if titleText is not None:
        title = np.zeros((40, image.shape[1], image.shape[2]))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # get boundary of this text
        textsize = cv2.getTextSize(titleText, font, 1, 2)[0]
        # get coords based on boundary
        textX = int((title.shape[1] - textsize[0]) / 2)
        textY = int((title.shape[0] + textsize[1]) / 2)
        # put text on the title image
        cv2.putText(title, titleText, (textX, textY ), font, 1, (255, 255, 255), 2)
        # stack title to image
        image = np.vstack([title, image])
    
    return image

# parsing arguments
parser = argparse.ArgumentParser(description='test script to verify we can sample trajectories from the environment.')
parser.add_argument('--dataroot', '-r',  type=str, help='path to the XCAT CT volumes.')
parser.add_argument('--volume_id', '-vol_id', type=str, default='default_512_CT_1', help='filename of the CT volume.')
parser.add_argument('--segmentation_id', '-seg_id', type=str, default='default_512_SEG_1', help='filename of the segmented CT volume.')
parser.add_argument('--Nsteps', '-N',  type=int, default=100, help='how many steps to take.')
parser.add_argument('--savedir', '-s', type=str, default='./trajectories/', help='where to save the trajectory.')
parser.add_argument('--name', '-n', type=str, default='sample', help='name of the experiment.')

args = parser.parse_args()

if __name__=="__main__":

    # instanciate environment
    env = BaseEnvironment(args.dataroot, args.volume_id, args.segmentation_id)

    # get the current plane
    state, reward, seg = env.sample()

    # append frame to array
    frames = [render_frame(state, seg, "reward: {}".format(reward))]

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
        state, reward, seg = env.step(*increments)

        # append frame to array
        frames.append(render_frame(state, seg, "reward: {}".format(reward)))

    
    # save all frames as a GIF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(os.path.join(args.savedir, args.name+'.gif'), fps=10)