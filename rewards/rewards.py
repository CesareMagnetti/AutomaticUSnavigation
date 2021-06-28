import numpy as np

class AnatomyReward(object):
    """Class to assign the anatomical reward signal.
    """
    def __init__(self, rewardIDs):
        # if more IDs are passed store in an array
        self.IDs = rewardIDs.split(",")
    
    def __call__(self, seg):
        """ evaluates the anotical reward as the ratio of pixels containing structures of interest in the segmented slice.
        Params:
        ==========
            seg (np.ndarray): segmented slice of the current state.
        returns -> float, ratio of pixels of interest wrt slice pixel count.
        """
        rewardAnatomy = 0
        for ID in seld.IDs:
            rewardAnatomy += (seg==ID).sum().item()
        rewardAnatomy/=np.prod(seg.shape)
        return rewardAnatomy

class SteppingReward(object):
    """Class to assign the default reward received upon making a step
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
    to encourage them to stay far away from each other and hence in a more meaningful part of the volume.
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

    def __call__(self, next_state):
        """ give a penalty proportional to how many pixels OOB the agent is. If the agent is within the volume then do not give any penalty.
        Params:
        ==========
            next_state (np.ndarray): the state the agent stepped into as 3 stacked 3D vectors.
        returns -> float, corresponding reward
        """
        total_penalty = 0
        for s in next_state:
            total_penalty+=self.penalty*max(0, s[0] - self.sx)
            total_penalty+=self.penalty*max(0, s[1] - self.sy)
            total_penalty+=self.penalty*max(0, s[2] - self.sz)      
        return total_penalty
