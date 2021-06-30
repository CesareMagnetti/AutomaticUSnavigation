from moviepy.editor import ImageSequenceClip
import numpy as np

class Visualizer():    
    def render_frames(self, frames, fname, fps=10):
        frames = [frame[..., np.newaxis]*np.ones(3) for frame in frames]
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(fname, fps=fps)

    def render_states(self, states, fname, fps=10):
        raise NotImplementedError()
    
    def render(self, out):
        raise NotImplementedError()



