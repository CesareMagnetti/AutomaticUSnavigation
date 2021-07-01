import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip

class Visualizer():    
    def render_frames(self, frames, fname, fps=10):
        frames = [frame[..., np.newaxis]*np.ones(3) for frame in frames]
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(fname, fps=fps)

def render_full(self, states, frames, fname):

        def update(num, states, frames, lines):
            lines[0].set_data(np.append(states[num,:,0], states[num,0,0]), np.append(states[num,:,1], states[num,0,1]))
            lines[0].set_3d_properties(np.append(states[num,:,2], states[num,0,2]))
            lines[0].set_marker("o")
            lines[2].set_data(frames[num])
            return lines

        # Attaching 3D axis to the figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122)

        # plot the first frame
        states = np.vstack([state[np.newaxis, ...] for state in states])
        lines = [ax.plot(np.append(states[0,:,0], states[0,0,0]), 
                         np.append(states[0,:,1], states[0,0,1]),
                         np.append(states[0,:,2], states[0,0,2]))[0],
                 plot_linear_cube(ax, 0, 0, 0, 256, 256, 256)[0],
                 ax1.imshow(frames[0], cmap="Greys_r")]

        ax.set_xlim(-50,300)
        ax.set_ylim(-50,300)
        ax.set_zlim(-50,300)
        # label the axis
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # titles
        ax.set_title("agents")
        ax1.set_title("sampled slice")

        plt.rcParams['animation.html'] = 'html5'
        line_ani = animation.FuncAnimation(fig, update, 10, fargs=(states, frames, lines), interval=100, blit=True, repeat=True)
        #plt.show()
        line_ani.save(fname, writer='imagemagick',fps=10)



