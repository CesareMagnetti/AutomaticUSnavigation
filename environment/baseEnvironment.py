from environment.utils import ReplayBuffer
import torch, os, six, random
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from abc import abstractmethod, ABCMeta
import torch.multiprocessing as mp
import concurrent.futures

@six.add_metaclass(ABCMeta)
class BaseEnvironment(object):
    """
    base class for all of our navigation environments objects
    """
    def __init__(self, config):
        """ Initialize environment object
        Params:
        =========
            config (argparse object): all useful options to build the environment (see options/options.py for details)
        """

        # setup data, checkpoints and results dirs for any logging/ input output
        self.dataroot = config.dataroot
        self.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
        self.results_dir = os.path.join(config.results_dir, config.name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        # set up the device
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        # set up the replay buffer
        self.buffer = ReplayBuffer(config.buffer_size, config.batch_size)
        # save the config for any options we might need
        self.config = config

    @abstractmethod
    def sample_plane(self, state, return_seg=False, oob_black=True):
        """ Universal function to sample a plane from 3 3D points
        """

    @abstractmethod
    def step(self, action):
        """Perform an input action, observe the next state and reward.
        Automatically stores the tuple (state, action, reward, next_state) to the replay buffer.
        """
    
    def reset(self):
        """Reset the environment at the end of an episode. It will set a random initial state for the next episode
        and reset all the logs that the environment is keeping.
        """
        raise NotImplementedError()
    
    def get_reward(self):
        """Calculates the corresponding reward of stepping into a new state.
        """
        raise NotImplementedError()
    
    def sample_planes(self, states, **kwargs):
        """Sample multiple queried planes launching multiple threads in parallel. This function is useful if the self.sample_plane()
        function is time consuming and we whish to query several planes. And example of this scenario is when we sample a batch from
        the replay buffer: it is impractical to store full image frames in the replay buffer, it is more efficient to only store the
        coordinates of the 3 points we need in order to sample the corresponding plane. This means that every time we train the Qnetwork
        we need to sample planes for a batch of states and next states. If sampling a single plane is costly, this will lead to an 
        extremely slow training.

        Params
        ==========
        states (list/tuple), all states that we wish to sample.

        returns -> planes (list), a list containing all sampled planes
        """
        # sample planes using multi-thread
        planes = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.sample_plane, state, **kwargs) for state in states]
        [planes.append(f.result()) for f in futures]
        return planes
    
    def random_walk(self, n_random_steps, n_random_restarts = 0, return_trajectory=False):
        """ Starts a random walk to gather observations (s, a, r, s').
        Will return the number of unique states reached by each agent to quantify the amount of exploration
        Params:
        ==========
            n_random_steps (int): number of steps for which we want the random walk to continue.
            n_random_restarts (int): number of times we reset the agent to a random position during the walk.
            return_trajectory (bool): if True we return the trajectory followed by the agent.
        """
        # get how many steps to do before each restart
        restart_freq = n_random_steps/(n_random_restarts+1)
        # get the trajectory if needed
        if return_trajectory:
            trajectory = []
        # start the random walk
        self.reset()
        for step in tqdm(range(1, n_random_steps+1), desc="random walk..."):
            # random action
            action = np.vstack([random.choice(np.arange(self.config.action_size)) for _ in range(self.config.n_agents)])
            # step the environment according to this random action (automatically stores (s, a, r ,s') to buffer)
            _ = self.step(action)
            # get the visual if needed
            if return_trajectory:
                trajectory.append(self.state)
            # restart environment if needed
            if step % restart_freq == 0:
                self.reset()
        if return_trajectory:
            return trajectory

            

    def render(self, states, with_seg=False, with_cube=False, fname="sample"):
        """Renders a single state or many states. If many states it will use multiple processes.
        Params:
        ==========
            states (np.ndarray] or list of such tuples): list of states to render.
            with_seg (bool): flag if to render the segmentations as well as the states.
            with_cube (bool): flag if to render the orientation of the sampled plane along with the plane itself.
            fname (str): name of the file we will save
        """
        # 1. sample the plane(s)
        if isinstance(states, np.ndarray):
            states = [states]
        slices = self.sample_planes(states, return_seg=with_seg)

        # 2. render each plane
        fig = plt.figure(0)
        for i, (state, slice) in enumerate(states, slices):
            if with_seg:
                slice, seg = slice
            else:
                seg = None

            # get the slice and segmentation corresponding to the current state
            if seg is not None:
                slice = slice.cpu().numpy().squeeze()*255 # un-normalize slice for plotting
                ax = fig.add_subplot(111)
                ax.imshow(slice, cmap="Greys")
                ax.set_title("CT slice")
                ax = fig.add_subplot(212)
                ax.imshow(seg, cmap="Greys")
                ax.set_title("seg slice")
            else:
                slice = slice.cpu().numpy().squeeze()
                ax = fig.add_subplot(111)
                ax.imshow(slice, cmap="Greys")
                ax.set_title("CT slice")
                
            if state is not None:
                if seg is not None:
                    ax = fig.add_subplot(313, projection='3d')
                else:
                    ax = fig.add_subplot(212, projection='3d')
                # plot the volume as a cube
                xx = [0, 0, self.sx, self.sx, 0]
                yy = [0, self.sy, self.sy, 0, 0]
                kwargs = {'alpha': 0.3, 'color': 'red'}
                ax.plot3D(xx, yy, [0]*5, **kwargs)
                ax.plot3D(xx, yy, [self.sz]*5, **kwargs)
                ax.plot3D([0, 0], [0, 0], [0, self.sz], **kwargs)
                ax.plot3D([0, 0], [self.sy, self.sy], [0, self.sz], **kwargs)
                ax.plot3D([self.sx, self.sx], [self.sy, self.sy], [0, self.sz], **kwargs)
                ax.plot3D([self.sx, self.sx], [0, 0], [0, self.sz], **kwargs)
                # plot the 2D slice on a 3D surface
                xx, yy = np.meshgrid(range(self.sx), range(self.sy))
                a,b,c,d = self.get_plane_coefs(*state)
                zz = (-d -a*xx -b*yy)/c
                ax.plot_surface(xx,yy,zz, alpha=0.7)
                ax.set_title("slice orientation")
            if not os.path.exists("./temp_frames"):
                os.makedirs("./temp_frames")
            plt.savefig("./temp_frames/%d.png"%i)
        
        # 3. make an animation with all successive frames or save the single frame passed
        if i>0:
            #create GIF with image magick convert function
            os.system('convert   -delay 4   -loop 0   ./temp_frames/*.png   animatedTrajectory.gif')


    @staticmethod
    def get_plane_coefs(p1, p2, p3):
        """ Gets the coefficients of a 3D plane given the coordinates of 3 3D points
        """
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        return a, b, c, d

    @staticmethod
    def mapActionToIncrement(action):
        """ Maps a discrete action to a specific increment that will be added to the state in self.step() in order
        to move towards the next state.

        It uses the following convention for 3D navigation:
        0: +1 in x coordinate
        1: -1 in x coordinate
        2: +1 in y coordinate
        3: -1 in y coordinate
        4: +1 in z coordinate
        5: -1 in z coordinate

        returns -> incr (np.ndarray) of shape: (3,). It contains a unit increment in a particular 3D direction.
        """
        if action == 0:
            incr = np.array([1, 0, 0])
        elif action == 1:
            incr = np.array([-1, 0, 0])
        elif action == 2:
            incr = np.array([0, 1, 0])
        elif action == 3:
            incr = np.array([0, -1, 0])
        elif action == 4:
            incr = np.array([0, 0, 1])
        elif action == 5:
            incr = np.array([0, 0, -1])
        else:
            raise ValueError('unknown action: %d, legal actions: [0, 1, 2, 3, 4, 5]'%action)
        
        return incr
    
    @staticmethod
    def mapIncrementToAction(incr):
        """ Maps a specific increment to a discrete action, performing the inverse mapping to self.mapActionToIncrement().
        This function is useful if we want to store discrete actions rather than the corresponding increments in the replay buffer
        Since in the QLearning update we need such discrete form of the action.

        It uses the following convention:
        0: +1 in x coordinate
        1: -1 in x coordinate
        2: +1 in y coordinate
        3: -1 in y coordinate
        4: +1 in z coordinate
        5: -1 in z coordinate

        returns -> action (int). It contains the discrete action of a particular 3D unit movement.
        """
        if incr[0] == 1:
            action = 0
        elif incr[0] == -1:
            action = 1
        elif incr[1] == 1:
            action = 2
        elif incr[1] == -1:
            action = 3
        elif incr[2] == 1:
            action = 4
        elif incr[2] == -1:
            action = 5
        else:
            raise ValueError('unknown increment: {}.'.format(incr))
        return action
