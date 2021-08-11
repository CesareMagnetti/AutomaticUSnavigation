import torch, os, six, random
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from abc import abstractmethod, ABCMeta
import matplotlib.gridspec as gridspec
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
        # save the config for any options we might need
        self.config = config

    @abstractmethod
    def sample_plane(self, state, return_seg=False, oob_black=True):
        """ Universal function to sample a plane from 3 3D points
        """

    @abstractmethod
    def step(self, action, buffer):
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
        samples = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.sample_plane, state, **kwargs) for state in states]
        [samples.append(f.result()) for f in futures]
        # convert list of dicts to dict of lists (assumes all dicts have same keys, which is our case)
        samples = {k: [dic[k] for dic in samples] for k in samples[0]}
        return samples
    
    def random_walk(self, n_random_steps, buffer = None, return_trajectory=False):
        """ Starts a random walk to gather observations (s, a, r, s').
        Will return the number of unique states reached by each agent to quantify the amount of exploration
        Params:
        ==========
            n_random_steps (int): number of steps for which we want the random walk to continue.
            buffer (buffer/* instance): if passed, a ReplayBuffer instance to collect memory.
            return_trajectory (bool): if True we return the trajectory followed by the agent.
        """
        # get the trajectory if needed
        if return_trajectory:
            trajectory = []
        # start the random walk
        self.reset()
        for step in range(1, n_random_steps+1):
            # random action
            action = np.vstack([random.choice(np.arange(self.config.action_size)) for _ in range(self.config.n_agents)])
            # step the environment according to this random action
            transition, next_slice = self.step(action)
            # add (state, action, reward, next_state) to buffer
            if buffer is not None:
                buffer.add(transition)
            # get the visual if needed
            if return_trajectory:
                trajectory.append(self.state)
            # restart environment after max steps per episode are reached
            if step % self.config.n_steps_per_episode == 0:
                self.reset()
        if return_trajectory:
            return trajectory

    def get_plane_from_points(self, points, shape):
            """ function to sample a plane from 3 3D points (state)
            Params:
            ==========
                points (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the volume.
                shape (np.ndarry): shape of the 3D volume to sample plane from

                returns -> (X,Y,Z) ndarrays that will index Volume in order to extract a specific plane.
            """
            # get plane coefs
            a,b,c,d = self.get_plane_coefs(*points)
            # get volume shape
            sx, sy, sz = shape
            # extract corresponding slice
            main_ax = np.argmax([abs(a), abs(b), abs(c)])
            if main_ax == 0:
                Y, Z = np.meshgrid(np.arange(sy), np.arange(sz), indexing='ij')
                X = (d - b * Y - c * Z) / a

                X = X.round().astype(np.int)
                P = X.copy()
                S = sx-1

                X[X <= 0] = 0
                X[X >= sx] = sx-1

            elif main_ax==1:
                X, Z = np.meshgrid(np.arange(sx), np.arange(sz), indexing='ij')
                Y = (d - a * X - c * Z) / b

                Y = Y.round().astype(np.int)
                P = Y.copy()
                S = sy-1

                Y[Y <= 0] = 0
                Y[Y >= sy] = sy-1
            
            elif main_ax==2:
                X, Y = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
                Z = (d - a * X - b * Y) / c

                Z = Z.round().astype(np.int)
                P = Z.copy()
                S = sz-1

                Z[Z <= 0] = 0
                Z[Z >= sz] = sz-1
            
            return (X,Y,Z), P, S

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

        # normalize the coeffs (they would still define the same plane)
        norm = np.sum([abs(a), abs(b), abs(c), abs(d)])
        a /= norm
        b /= norm
        c /= norm
        d /= norm

        return np.array([a, b, c, d])
        
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
        6: stays still/does not lead to an increment

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
        elif action == 6:
            incr = np.array([0, 0, 0])
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
        6: stays still/does not lead to an increment

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
        elif incr[0] == 0 and incr[1] == 0 and incr[2] == 0:
            action = 6
        else:
            raise ValueError('unknown increment: {}.'.format(incr))
        return action