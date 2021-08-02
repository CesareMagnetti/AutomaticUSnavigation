import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

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
    def __init__(self, savedir):
        self.savedir = os.path.join(savedir, "visuals")
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        
    def render_frames(self, frames, fname, fps=10):
        # images could be CxHxW if --location_aware, retain only the first channel.
        if len(frames[0].shape) > 2:
            # extract anatomy planes
            planes = [elem[0, ...] for elem in frames]
            planes = [elem[..., np.newaxis]*np.ones(3)*255 for elem in planes]
            # extract location maps
            locs1 = [elem[1, ...]*255 for elem in frames]
            locs1 = [elem[..., np.newaxis]*np.ones(3)*255 for elem in locs1]
            locs2 = [elem[2, ...]*255 for elem in frames]
            locs2 = [elem[..., np.newaxis]*np.ones(3)*255 for elem in locs2]
            locs3 = [elem[3, ...]*255 for elem in frames]
            locs3 = [elem[..., np.newaxis]*np.ones(3)*255 for elem in locs3]
            # stack the anatomy planes to the location maps horizontally for each frame
            frames = [np.hstack([plane,loc1,loc2,loc3]) for plane,loc1,loc2,loc3 in zip(planes,locs1,locs2,locs3)]
        else:
            frames = [elem[..., np.newaxis]*np.ones(3)*255 for elem in frames]
        # generate the gif
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(os.path.join(self.savedir, fname), fps=fps)

    def render_full(self, out, fname, fps=10):

            def update(num, states, frames, logs, plot_objects):
                plot_objects[0].set_data(np.append(states[num,:,0], states[num,0,0]), np.append(states[num,:,1], states[num,0,1]))
                plot_objects[0].set_3d_properties(np.append(states[num,:,2], states[num,0,2]))
                plot_objects[1]._offsets3d = (states[num,:,0], states[num,:,1], states[num,:,2])
                plot_objects[2].set_text("oob: {:.2f}".format(logs["oobReward_1"][num]))
                plot_objects[3].set_text("oob: {:.2f}".format(logs["oobReward_2"][num]))  
                plot_objects[4].set_text("oob: {:.2f}".format(logs["oobReward_3"][num]))
                plot_objects[5].set_text("area reward: {:.4f}".format(logs["areaReward"][num]))
                plot_objects[7].set_data(frames[num])                             
                plot_objects[8].set_text("anatomy reward: {:.4f}".format(logs["anatomyReward"][num]))
                for plot, log in zip(plot_objects[9], logs.values()):
                    plot.set_data(range(len(log[:num])), log[:num]/(abs(log).max()+10e-6)) # normalize for ease of visualization
                return plot_objects

            # gather useful information
            # 1. rearrange logs
            logs = {key: [] for key in out["logs"][0]}
            for log in out["logs"]:
                for key, item in log.items():
                    logs[key].append(item)
            for key in logs:
                logs[key] = np.array(logs[key])
            # 2. stack the states in a single numpy array
            states = np.vstack([state[np.newaxis, ...] for state in out["states"]])
            # 3. process theframes (the slice will always be the 0th channel in the image)
            # images could be CxHxW if location maps are passed.
            if len(out["frames"][0].shape) == 2:
                frames = out["frames"]
            elif len(out["frames"][0].shape) == 3:
                frames = [elem[0, ...] for elem in out["frames"]]
            else:
                raise ValueError("entries in out['frames'] have wrong dimensionality.")

            # Attaching 3D axis to the figure
            fig = plt.figure(figsize=(14, 4))
            ax = fig.add_subplot(131, projection='3d')
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)   
            # instanciate a list of plot objects that will be dynamically updated to create an animation
            plot_objects = [# 1. render the current position of the agents
                            ax.plot(np.append(states[0,:,0], states[0,0,0]), 
                                    np.append(states[0,:,1], states[0,0,1]),
                                    np.append(states[0,:,2], states[0,0,2]), c='k', ls="--", alpha=0.5)[0],
                            ax.scatter(states[0,:,0],states[0,:,1],states[0,:,2], color=['r', 'g', 'b']),
                            # 2. render a legend with the current rewards of the agents for being within the volume boundaries
                            ax.text2D(0.2, 0.95, "oob: ", color='red', transform=ax.transAxes),
                            ax.text2D(0.5, 0.95, "oob: ", color='green', transform=ax.transAxes),
                            ax.text2D(0.8, 0.95, "oob: ", color='blue', transform=ax.transAxes),
                            # add the current reward the agents receive for distancing from each other (not clustering at a point)
                            ax.text2D(0.3, 1, "area reward: ", color='black', transform=ax.transAxes),
                            # 3. plot the boundaries of the volume as a reference
                            plot_linear_cube(ax, 0, 0, 0, 256, 256, 256)[0],
                            # 4. plot the current imaged slice on the second subplot
                            ax1.imshow(frames[0], cmap="Greys_r"),
                            # add the current reward given the context anatomy contained in the slice
                            ax1.text(0.25, 1, "anatomy reward: ", color='black', transform=ax1.transAxes),
                            # 5. running plots for all losses
                            [ax2.plot([], [], label=''.join(key.lower().split("reward")))[0] for key in logs],
                            ]

            # set the limits for the axes (slightly larger than the volume to observe oob points)
            ax.set_xlim(-50,300)
            ax.set_ylim(-50,300)
            ax.set_zlim(-50,300)
            # label the axis
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # set the legend for the lineplots
            ax2.legend(bbox_to_anchor=(1.4, 1), ncol = 1, fontsize=10)
            ax2.set_xlim(0, states.shape[0])
            ax2.set_ylim(-1,1)
            ax2.set_title("collected rewards")
            ax2.set_xlabel("steps")
            ax2.set_ylabel("normalized reward")
            plt.rcParams['animation.html'] = 'html5'
            line_ani = animation.FuncAnimation(fig, update, len(states), fargs=(states, frames, logs, plot_objects))
            line_ani.save(fname, fps=fps)

    def render_frames_with_segmentations(self, planes, segs, fname, fps=10):
        frames = [np.hstack([plane, seg]) for plane, seg in zip(planes, segs)]
        frames = [elem[..., np.newaxis]*np.ones(3)*255 for elem in frames]
        # generate the gif
        clip = ImageSequenceClip(frames, fps=fps)
        print(os.path.join(self.savedir, fname))
        clip.write_gif(os.path.join(self.savedir, fname), fps=fps)


