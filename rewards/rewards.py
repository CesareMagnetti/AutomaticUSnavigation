import numpy as np
from collections import deque, Counter

class PlaneDistanceReward(object):
    """Class to assign a reward based on the plane distance between the plane defined by the current state and the goal plane
    defined using the centroids of the heart chambers.
    """
    def __init__(self, goal):
        self.goal = goal
        self.previous_plane = None
    
    def __call__(self, coefs):
        # if previous plane is none (first step) set it equal to the current step -> reward of zero at the beginning
        if self.previous_plane is None:
            raise ValueError('self.previous_plane not defined. Overwrite it as ``instance.previous_plane = self.get_plane_coefs(pointA, pointB, pointC)``')
        # calculate euclidean distance between current plane and the goal
        D1 = ((coefs-self.goal)**2).sum()
        # calculate distance between previous plane and goal
        D2 = ((self.previous_plane-self.goal)**2).sum()
        # store plane as the new previous plane
        self.previous_plane = np.array(coefs)
        # return sign function of distance improvement (D1 should be smaller than D2 if we are getting closer -> +1 if closer, -1 if further, 0 if same distance)
        return np.sign(D2-D1)
        # return distance improvement (D1 should be smaller than D2 if we are getting closer)
        #return D2-D1

class Oscillate(object):
    def __init__(self, history_length, stop_freq):
        self.history = deque(maxlen=history_length)
        self.stop_freq = stop_freq
    
    def __call__(self, state):
        # append new state
        self.history.append(state)
        # get frequency of seen states in hystory
        counter = Counter(self.history)
        freq = counter.most_common()
        # if most common plane seen more than stop_freq, then stop
        if freq[0][1] > self.stop_freq:
            return True
        else:
            return False

class AnatomyReward(object):
    """Class to assign the anatomical reward signal. we reward based on a particular anatomical structure being present
    in the slice sampled by the current state to this end we have segmentations of the volume and reward_id corresponds
    to the value of the anatomical tissue of interest in the segmentation (i.e. the ID of the left ventricle is 2885).
    """
    def __init__(self, rewardIDs, is_penalty=False, incremental=False, weight=1):
        # if more IDs are passed store in an array
        self.IDs = [int(ID) for ID in rewardIDs.split(",")]
        self.is_penalty = is_penalty
        self.incremental = incremental
        self.weight = weight
        if incremental:
            self.previous_reward = None

    def get_anatomy_reward(self, seg):
        rewardAnatomy = 0
        for ID in self.IDs:
            rewardAnatomy += (seg==ID).sum().item()
        rewardAnatomy/=np.prod(seg.shape)
        if self.is_penalty:
            rewardAnatomy*=-1
        return rewardAnatomy

    def __call__(self, seg):
        """ evaluates the anotical reward as the ratio of pixels containing structures of interest in the segmented slice.
        Params:
        ==========
            seg (np.ndarray): segmented slice of the current state.
        returns -> float, ratio of pixels of interest wrt slice pixel count.
        """
        if not self.incremental:
            return self.get_anatomy_reward(seg)
        else:
            # the current slice should have higher anatomical reward if we are getting closer -> +1 if more anatomical content, -1 if less, 0 if same
            current_reward = self.get_anatomy_reward(seg)
            reward = np.sign(current_reward - self.previous_reward)
            self.previous_reward = current_reward
            return self.weight*reward # scale by weight

class SteppingReward(object):
    """Class to assign the default reward received upon making a step. we give a small penalty for each step in which the above ID
    is not present in the sampled slice. As soon as even one pixel of the structure of interest enters the sampled slice, we stop 
    the penalty. Like this the agent is incetivized to move towards the region of interest quickly.
    """
    def __init__(self, penalty):
        self.penalty = -abs(penalty)
    
    def __call__(self, give_penalty):
        """ Give a static penalty after each step if give_penalty flag is True, else do not reward nor penalyze.
        This should incentivize the agent to move towards the goal quicker as when the agent is close to the goal, the give_penalty
        flag will be set to False.
        Params:
        =========
            give_penalty (bool): flag if penalize or not
        returns -> float, the amount of penalty the agent will receive.
        """
        if give_penalty:
            return self.penalty
        else:
            return 0

class AreaReward(object):
    """ In order to prevent the agents from clustering close together, we reward proportionally to the area of the triangle spanned by the 3 agents,
    to encourage them to stay far away from each other and hence in a more meaningful part of the volume. Arguably this should also embed the agent
    with some prior information about the fact that they are sampling a plane and should work together (with a shared objective) rather than indepentently.
    """
    def __init__(self, weight, max_area):
        self.weight = weight
        self.max_area = max_area
    
    def __call__(self, state):
        """ Takes the current state (3 stacked 3D points), and evaluates the corresponding reward (proportional to area of triangle)
        Params:
        ==========
            state (np.ndarray): 3 stacked 3D points organized in a 2D np array.
        return -> float, corresponding reward
        """
        area = self.get_traingle_area(*state)
        area/=self.max_area
        return self.weight*area
    
    @staticmethod
    def get_traingle_area(a, b, c) :
        return 0.5*np.linalg.norm(np.cross(b-a, c-a))

class OutOfBoundaryReward(object):
    """ Reward an agent receives upon stepping outside of the current volume. A plane can still be sampled if the agents move outside of
    the volume, however we believe its sensible to incentivize the agents to stay within the Volume boundaries by penalizying OOB points.
    """
    def __init__(self, penalty, sx, sy, sz):
        self.penalty = -abs(penalty)
        self.sx, self.sy, self.sz = sx, sy, sz

    def __call__(self, point):
        """ give a penalty proportional to how many pixels OOB the agent is. If the agent is within the volume then do not give any penalty.
        Params:
        ==========
            point (np.ndarray): the current location of an agent (3 element array).
        returns -> float, corresponding oob reward for the agent
        """
        total_penalty = 0
        for coord, s in zip(point, [self.sx, self.sy, self.sz]):
            # the origin of the volume is at (0,0,0), hence the agent is out of boundary
            # if its position is >= than the resolution of the volume OR if its position is
            # negative. We count the number of pixels out of boundary for each dimension as a penalty,
            # scaled by self.penalty
            if coord >= 0:
                total_penalty+=self.penalty*max(0, coord-s)
            else:
                total_penalty+=self.penalty*abs(coord) 
        return total_penalty

class StopReward(object):
    """Rewards the agents receive when they stop (all agents output the action to stay still). When acting greedily this means that
       that the navigation will stop on the current frame. If the frame is bad (far from the goal_reward) then give a penalty to the
       agents (or a reward if doing better than the goal).
    """
    def __init__(self, scale, goal_reward):
        self.scale = abs(scale)
        self.goal_reward = goal_reward
    
    def __call__(self, increment, current_reward):
        # if all agents choose to not move, then increment will be an all-zero array,
        # in this case increment.any() will return False and we give the penalty.
        if increment.any():
            return 0
        else:
            # get difference in reward from current step and the goal reward
            D = current_reward-self.goal_reward # negative if doing worse than goal, positive if doing better than goal 
            # scale reward as queried
            return self.scale*D

