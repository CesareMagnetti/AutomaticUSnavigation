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
                plot_objects[2].set_text("oob: {:.2f}".format(logs[num]["oobReward_1"]))
                plot_objects[3].set_text("oob: {:.2f}".format(logs[num]["oobReward_2"]))  
                plot_objects[4].set_text("oob: {:.2f}".format(logs[num]["oobReward_3"])) 
                plot_objects[5].set_data(frames[num])                             
                plot_objects[7].set_title('cumulative areaReward: {:.4f}'.format(logs[num]["areaReward"]))
                plot_objects[8].set_title("cumulative anatomy reward: {:.4f}".format(logs[num]["anatomyReward"]))
                return plot_objects

            states, frames, logs = out["states"], out["frames"], out["logs"]
            # Attaching 3D axis to the figure
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(121, projection='3d')
            ax1 = fig.add_subplot(122)

            # plot the first frame
            states = np.vstack([state[np.newaxis, ...] for state in states])
            plot_objects = [ax.plot(np.append(states[0,:,0], states[0,0,0]), 
                                    np.append(states[0,:,1], states[0,0,1]),
                                    np.append(states[0,:,2], states[0,0,2]), c='k', ls="--", alpha=0.5)[0],
                            ax.scatter(states[0,:,0],states[0,:,1],states[0,:,2], color=['r', 'g', 'b']),
                            ax.text2D(0.2, 0.95, "oob: {:.2f}".format(logs[0]["oobReward_1"]), color='red', transform=ax.transAxes),
                            ax.text2D(0.5, 0.95, "oob: {:.2f}".format(logs[0]["oobReward_2"]), color='green', transform=ax.transAxes),
                            ax.text2D(0.8, 0.95, "oob: {:.2f}".format(logs[0]["oobReward_3"]), color='blue', transform=ax.transAxes),
                            ax1.imshow(frames[0], cmap="Greys_r"),
                            plot_linear_cube(ax, 0, 0, 0, 256, 256, 256)[0],
                            ax,
                            ax1]
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



