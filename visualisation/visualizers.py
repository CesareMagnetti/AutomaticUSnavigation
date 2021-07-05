import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip

def plot_linear_cube(ax, x, y, z, dx, dy, dz, color='black'):
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    return ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)

class Visualizer():    
    def render_frames(self, frames, fname, fps=10):
        frames = [frame[..., np.newaxis]*np.ones(3) for frame in frames]
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(fname, fps=fps)

    def render_full(self, out, fname, fps=10):

            def update(num, states, frames, logs, plot_objects):
                plot_objects[0].set_data(np.append(states[num,:,0], states[num,0,0]), np.append(states[num,:,1], states[num,0,1]))
                plot_objects[0].set_3d_properties(np.append(states[num,:,2], states[num,0,2]))
                plot_objects[1]._offsets3d = (states[num,:,0], states[num,:,1], states[num,:,2])
                plot_objects[2].set_text("oob: {:.2f}".format(logs["oobReward_1"][num]))
                plot_objects[3].set_text("oob: {:.2f}".format(logs["oobReward_2"][num]))  
                plot_objects[4].set_text("oob: {:.2f}".format(logs["oobReward_3"][num])) 
                plot_objects[5].set_text("area reward: {:.2f}".format(logs["areaReward"][num]))
                plot_objects[7].set_data(frames[num])                             
                plot_objects[8].set_text("anatomy reward: {:.4f}".format(logs["anatomyReward"][num]))
                for plot, log in zip(plot_objects[9], logs):
                    plot.set_data(np.arange(len(logs[log][:num])), logs[log][:num])

                return plot_objects

            # gather useful information
            states, frames = out["states"], out["frames"]
            logs = {key: [] for key in out["logs"][0]}
            for log in out["logs"]:
                for key, item in log.items():
                    logs[key].append(item)

            # Attaching 3D axis to the figure
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)

            # stack the states in a single numpy array
            states = np.vstack([state[np.newaxis, ...] for state in states])

            # instanciate a list of plot objects that will be dynamically updated to create an animation
            plot_objects = [# 1. render the current position of the agents
                            ax.plot(np.append(states[0,:,0], states[0,0,0]), 
                                    np.append(states[0,:,1], states[0,0,1]),
                                    np.append(states[0,:,2], states[0,0,2]), c='k', ls="--", alpha=0.5)[0],
                            ax.scatter(states[0,:,0],states[0,:,1],states[0,:,2], color=['r', 'g', 'b']),
                            # 2. render a legend with the current rewards of the agents for being within the volume boundaries
                            ax.text2D(0.2, 0.95, "oob: {:.2f}".format(logs["oobReward_1"][0]), color='red', transform=ax.transAxes),
                            ax.text2D(0.5, 0.95, "oob: {:.2f}".format(logs["oobReward_2"][0]), color='green', transform=ax.transAxes),
                            ax.text2D(0.8, 0.95, "oob: {:.2f}".format(logs["oobReward_3"][0]), color='blue', transform=ax.transAxes),
                            # add the current reward the agents receive for distancing from each other (not clustering at a point)
                            ax.text2D(0.5, 1, "area reward: {:.2f}".format(logs["areaReward"][0]), color='black', transform=ax.transAxes),
                            # 3. plot the boundaries of the volume as a reference
                            plot_linear_cube(ax, 0, 0, 0, 256, 256, 256)[0],
                            # 4. plot the current imaged slice on the second subplot
                            ax1.imshow(frames[0], cmap="Greys_r"),
                            # add the current reward given the context anatomy contained in the slice
                            ax1.text(0.5, 1, "anatomy reward: {:.4f}".format(logs["anatomyReward"][0]), color='black', transform=ax.transAxes),
                            # 5. running plots for all losses
                            [ax2.plot(log[0], label=key)[0] for key,log in logs.items()],
                            ]
            # set the limits for the axes (slightly larger than the volume to observe oob points)
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
            line_ani = animation.FuncAnimation(fig, update, len(states), fargs=(states, frames, logs, plot_objects))
            line_ani.save(fname, fps=fps)



