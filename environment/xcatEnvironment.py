from environment.baseEnvironment import BaseEnvironment
from environment.utils import get_plane_from_points
from rewards.rewards import *
import numpy as np
import SimpleITK as sitk
import os
from moviepy.editor import ImageSequenceClip

class SingleVolumeEnvironment(BaseEnvironment):
    def __init__(self, config, vol_id=0):
        """
        Initialize an Environment object, the environment will focuse on a single XCAT/CT volume.
    
        Params that we will use:
        ======
            config (argparse object): contains all options. see ./options/options.py for more details
            vol_id (int): specifies which volume we want to load. Specifically ``config.volume_ids`` should contain
                          all queried volumes by the main.py script. (i.e. parser.volume_ids = 'samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7')
                          we will load the volume at position ``vol_id`` (default=0)               
        """

        # initialize the base environment
        BaseEnvironment.__init__(self, config)

        # identify the volume used by this environment instance
        volume_ids = config.volume_ids.split(',')
        self.vol_id = volume_ids[vol_id]
        
        # load queried CT volume
        itkVolume = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_1_CT.nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume) 
        # preprocess volume
        if not config.no_preprocess:
            Volume = Volume/Volume.max()*255
            Volume = intensity_scaling(Volume, pmin=config.pmin, pmax=config.pmax, nmin=config.nmin, nmax=config.nmax)
        self.Volume = Volume.astype(np.uint8) 
        self.sx, self.sy, self.sz = self.Volume.shape 

        # load queried CT segmentation
        itkSegmentation = sitk.ReadImage(os.path.join(self.dataroot, self.vol_id+"_1_SEG.nii.gz"))
        Segmentation = sitk.GetArrayFromImage(itkSegmentation)
        self.Segmentation = Segmentation

        # load a black cube to be popoulated with the position of the agent (state). Slicing through this cube
        # will yield an image with the relative position of each agent within the slice.
        self.black_cube = np.zeros((self.sx, self.sy, self.sz))

        # setup the reward function arguments:
        self.rewards = {"anatomyReward": AnatomyReward(config.anatomyRewardIDs),
                        "steppingReward": SteppingReward(config.steppingReward),
                        "areaReward": AreaReward(config.areaRewardWeight, self.sx*self.sy),
                        "oobReward":OutOfBoundaryReward(config.oobReward, self.sx, self.sy, self.sz)}
        
        # get starting configuration
        self.reset()

    def sample_plane(self, state, return_seg=False, return_pos=False, oob_black=True):
        """ function to sample a plane from 3 3D points (state)
        Params:
        ==========
            state (np.ndarray of shape (3,3)): v-stacked 3D points that will define a particular plane in the CT volume.
            return_seg (bool): flag if we wish to return the corresponding segmentation map. (default=False)
            return_pos (bool): flag if we want to return a 2D image embedding of the positions of the agents. (default=False)
            oob_black (bool): flack if we wish to mask out of volume pixels to black. (default=True)

            returns -> plane (torch.tensor of shape (1, 1, self.sy, self.sx)): corresponding plane sampled from the CT volume (normalized and unsqueezed to 4D)
                       seg (optional, np.ndarray of shape (self.sy, self.sx)): segmentation map of the sampled plane
        """
        # 1. extract plane specs
        XYZ, P, S = get_plane_from_points(state, (self.sx, self.sy, self.sz))
        # 2. sample plane from the current volume
        X,Y,Z = XYZ
        plane = self.Volume[X,Y,Z]
        # mask out of boundary pixels to black
        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        # 3. sample the segmentation if needed
        if return_seg:
            seg=self.Segmentation[X,Y,Z]
        # 4. sample the 2D relative positions of the agents if needed
        if return_pos:
            # populate the black_box with the current state
            for s in state:
                if s[0]<self.sx and s[1]<self.sy and s[2]<self.sz:
                    self.black_cube[s[0],s[1],s[2]] = 1
            # sample the slice with agents positions
            pos=self.black_cube[X,Y,Z]
            # reset the black box to fully black
            for s in state:
                if s[0]<self.sx and s[1]<self.sy and s[2]<self.sz:
                    self.black_cube[s[0],s[1],s[2]] = 0
        
        if return_seg and return_pos:
            return plane, seg, pos
        elif return_seg:
            return plane, seg
        elif return_pos:
            return plane, pos
        else:
            return plane

    def get_reward(self, seg, state):
        """Calculates the corresponding reward of stepping into a state given its segmentation map.
        Params:
        ==========
            seg (np.ndarray of shape (self.sy, self.sx)): segmentation map from which to extract the reward.
            state (np.ndarray): 3 stacked 3D arrays representing the coordinates of the agent.
        Returns -> rewards (dict): the corresponding rewards collected by the agent.
        """
        rewards = {}
        # 1 reward given that we are slicing the right anatomical content.
        rewards["anatomyReward"] = self.rewards["anatomyReward"](seg)
        # 2 reward given upon stepping in a new state (only if there is no content of interest in the slice).
        rewards["steppingReward"] = self.rewards["steppingReward"](not rewards["anatomyReward"]>0)
        # 3 reward agents to be spreaded out rather homogeneously across the volume (not clustered).
        rewards["areaReward"] = self.rewards["areaReward"](state)
        # 4 give penalties when the agents move outside the volume.
        rewards["oobReward"] = self.rewards["oobReward"](state)
        return rewards

    def step(self, action):
        """Perform an input action (discrete action), observe the next state and reward.
        Params:
        ==========
            action (int): discrete action (see baseEnvironment.mapActionToIncrement())
        """
        # get the increment corresponding to this action
        increment = np.vstack([self.mapActionToIncrement(act) for act in action])
        # step into the next state
        state = self.state
        next_state = state + increment
        # observe the next plane and get the reward from segmentation map
        next_slice, segmentation = self.sample_plane(state=next_state, return_seg=True)
        rewards = self.get_reward(segmentation, next_state)
        # log these rewards to the current episode count
        for r in rewards:
            self.logs[r]+=rewards[r]
        # store the state in the visited states throughout the episode
        self.visited_states.append(next_state)
        # update the current state
        self.state = next_state
        # return transition and the next_slice which has already been sampled
        return (state, action, sum(rewards.values()), next_state), next_slice
    
    def reset(self):
        # sample a random plane (defined by 3 points) to start the episode from
        if self.config.easy_objective:
            # these planes correspond more or less to a 4-chamber view
            pointA = np.array([np.random.uniform(low=0.85, high=1)*self.sx,
                              0,
                              np.random.uniform(low=0.7, high=0.92)*self.sz])

            pointB = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                              self.sy,
                              0])

            pointC = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                              self.sy,
                              self.sz]) 
        else:
            pointA = np.array([np.random.uniform(low=0., high=1)*self.sx,
                              0,
                              np.random.uniform(low=0., high=1.)*self.sz])

            pointB = np.array([np.random.uniform(low=0., high=1.)*self.sx,
                              self.sy,
                              np.random.uniform(low=0., high=1.)*self.sz])

            pointC = np.array([np.random.uniform(low=0., high=1.)*self.sx,
                              self.sy,
                              np.random.uniform(low=0., high=1.)*self.sz])             
        # stack points to define the state
        self.state = np.vstack([pointA, pointB, pointC]).astype(np.int)
        # reset the logged rewards for this episode
        self.logs = {r: 0 for r in self.rewards}
        # reset the visited states throughout the episode
        self.visited_states = [self.state]

    def render_current_trajectory(self, fname, with_seg=False, with_pos=True, fps=10):
        # sample all needed planes along with needed segmentations or positions (multiple threads are launched)
        results = self.sample_planes(self.visited_states, return_seg=with_seg, return_pos=with_pos)
        # collect respective outputs
        if with_seg and with_pos:
            planes, segs, poss = list(zip(*results))
        elif with_seg:
            planes, segs = list(zip(*results))
            poss = [None]*len(planes)
        elif with_pos:
            planes, poss = list(zip(*results))
            segs = [None]*len(planes)
        else:
            planes = results
            segs, poss = [None]*len(planes), [None]*len(planes)
        
        # build the rendered animation
        frames = []
        for plane, seg, pos in zip(planes, segs, poss):
            # convert to RGB and compose the single frame
            IMAGE = plane[..., np.newaxis]*np.ones(3)
            if pos is not None:
                pos = pos[..., np.newaxis]*np.ones(3)
                IMAGE = np.hstack([IMAGE, pos])
            if seg is not None:
                seg = seg[..., np.newaxis]*np.ones(3)
                IMAGE = np.hstack([IMAGE, seg])
            frames.append(IMAGE)
        # render the clip as a gif
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(fname, fps=fps)

# ==== HELPER FUNCTIONS ====

def intensity_scaling(ndarr, pmin=None, pmax=None, nmin=None, nmax=None):
    pmin = pmin if pmin != None else ndarr.min()
    pmax = pmax if pmax != None else ndarr.max()
    nmin = nmin if nmin != None else pmin
    nmax = nmax if nmax != None else pmax
    
    ndarr[ndarr<pmin] = pmin
    ndarr[ndarr>pmax] = pmax
    ndarr = (ndarr-pmin)/(pmax-pmin)
    ndarr = ndarr*(nmax-nmin)+nmin
    return ndarr