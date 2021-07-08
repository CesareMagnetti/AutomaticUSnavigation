import numpy as np

class AnatomyReward(object):
    """Class to assign the anatomical reward signal. we reward based on a particular anatomical structure being present
    in the slice sampled by the current state to this end we have segmentations of the volume and reward_id corresponds
    to the value of the anatomical tissue of interest in the segmentation (i.e. the ID of the left ventricle is 2885).
    """
    def __init__(self, rewardIDs):
        # if more IDs are passed store in an array
        self.IDs = [int(ID) for ID in rewardIDs.split(",")]
    
    def __call__(self, seg):
        """ evaluates the anotical reward as the ratio of pixels containing structures of interest in the segmented slice.
        Params:
        ==========
            seg (np.ndarray): segmented slice of the current state.
        returns -> float, ratio of pixels of interest wrt slice pixel count.
        """
        rewardAnatomy = 0
        for ID in self.IDs:
            rewardAnatomy += (seg==ID).sum().item()
        rewardAnatomy/=np.prod(seg.shape)
        return rewardAnatomy

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
    """Rewards the agents receive when they stop (all agents output the action to stay still). When axting greedily this means that
       that the navigation will stop on the current frame. If the frame is bad (no anatomical context of interest contained in the image)
       then give a large penalty to the agents.
    """
    def __init__(self, penalty):
        self.penalty = -abs(penalty)
    
    def __call__(self, increment, give_penalty):
        # if all agents choose to not move, then increment will be an all-zero array,
        # in this case increment.any() will return False and we give the penalty.
        # give penalty will be a flag that the current frame does not contain any anatomy of interest.
        if increment.any():
            return 0
        else:
            if give_penalty:
                return self.penalty
            else:
                return 0
