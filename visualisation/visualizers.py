from utils import convertCoordinates3Dto2D
from moviepy.editor import ImageSequenceClip
import numpy as np

class Visualizer():    
    def render_frames(self, frames, fname, fps=10):
        frames = [frame[..., np.newaxis]*np.ones(3) for frame in frames]
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(fname, fps=fps)

    def render_states(self, states, volume_shape=None, fname, fps=10):
        # convert all points to 2D
        states2D = []
        for state in states:
            state2D = convertCoordinates3Dto2D(*state)
            states2D.append(state2D)
            print("3D state: {}\n2D state: {}".format(state, state2D))
    
    def render(self, out):
        raise NotImplementedError()



