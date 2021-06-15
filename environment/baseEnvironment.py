import numpy as np
import SimpleITK as sitk
import os
import torch
import cv2

class BaseEnvironment():
    def __init__(self, parser, **kwargs):
        """Initialize an Environment object.
        
        Params that we will use:
        ======
            parser.dataroot (str/pathlike): path to the CT volumes.
            parser.volume_id (str): specific id of the CT volume we intend to use.
            parser.no_scale_intensity (bool): flag to not scale the CT volume intensities.
            parser.use_cuda (bool): flag to use gpu or not.
            parser.reward_id (int): ID corresponding to the anatomical structure of interest in the segmentation.
        """
        # load CT volume
        itkVolume = sitk.ReadImage(os.path.join(parser.dataroot, parser.volume_id+"_CT_1.nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume)
        self.sx, self.sy, self.sz = Volume.shape

        # load CT segmentation
        itkSegmentation = sitk.ReadImage(os.path.join(parser.dataroot, parser.volume_id+"_SEG_1.nii.gz"))
        Segmentation = sitk.GetArrayFromImage(itkSegmentation)

        # preprocess volume
        Volume = Volume/Volume.max()*255
        if not parser.no_scale_intensity:
            Volume = intensity_scaling(Volume, pmin=kwargs.pop('pmin', 150), pmax=kwargs.pop('pmax', 200),
                                       nmin=kwargs.pop('nmin', 0), nmax=kwargs.pop('nmax', 255))
        
        # get the devce for gpu dependencies
        self.device = torch.device('cuda' if parser.use_cuda else 'cpu')

        # save CT volume and segmentation as class attributes
        self.Volume = torch.tensor(Volume, requires_grad=False, dtype=torch.uint8).to(self.device)
        self.Segmentation = torch.tensor(Segmentation, requires_grad=False).to(self.device)

        # ID of the segmentation corresponding to the anatomical structure we are trying to observe.
        self.rewardID = parser.reward_id

        # get starting configuration
        # we do not set a seed here, so that each time we call reset, the agent will find itself 
        # in a different plane, prompting for exploring starts. However, the reset function put a
        # constraint on the set of possible starting planes (we do not want a completely random/meaningless plane)
        self.reset()

    def sample(self, state, return_seg=False, oob_black=True):
        # get plane coefs
        a,b,c,d = get_plane_coefs(*state)

        # extract corresponding slice
        main_ax = np.argmax([abs(a), abs(b), abs(c)])
        if main_ax == 0:
            Y, Z = np.meshgrid(np.arange(self.sy), np.arange(self.sz), indexing='ij')
            X = (d - b * Y - c * Z) / a

            X = X.round().astype(np.int)
            P = X.copy()
            S = self.sx-1

            X[X <= 0] = 0
            X[X >= self.sx] = self.sx-1

        elif main_ax==1:
            X, Z = np.meshgrid(np.arange(self.sx), np.arange(self.sz), indexing='ij')
            Y = (d - a * X - c * Z) / b

            Y = Y.round().astype(np.int)
            P = Y.copy()
            S = self.sy-1

            Y[Y <= 0] = 0
            Y[Y >= self.sy] = self.sy-1
        
        elif main_ax==2:
            X, Y = np.meshgrid(np.arange(self.sx), np.arange(self.sy), indexing='ij')
            Z = (d - a * X - b * Y) / c

            Z = Z.round().astype(np.int)
            P = Z.copy()
            S = self.sz-1

            Z[Z <= 0] = 0
            Z[Z >= self.sz] = self.sz-1
        
        plane = self.Volume[X, Y, Z]

        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        
        if return_seg:
            return plane, self.Segmentation[X, Y, Z]
        else:
            return plane

    def get_reward(self, seg):
        # sample the according reward (i.e. count of pixels in left ventricle)
        # the left ventricle ID in the segmentation is 2885. let's count the number
        # of pixels in the left ventricle as a fit function, the agent will have to
        # find a view that maximizes the amount of left ventricle present in the image.
        reward = (seg==self.rewardID).sum().item()
        reward/=np.prod(seg.shape) # normalize by all pixels count
        return reward

    def step(self, increment):
        # update current state
        self.state+=increment
        # observe the next plane and get the reward (from segmentation map)
        _, segmentation = self.sample(state=self.state, return_seg=True)
        reward = self.get_reward(segmentation)
        return self.state, reward
    
    def reset(self):
        # get a meaningful starting plane (these hardcoded ranges will yield views close to 4-chamber view) 
        pointA = np.array([np.random.uniform(low=0.85, high=1)*self.sx,
                           0,
                           np.random.uniform(low=0.7, high=0.92)*self.sz])

        pointB = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                           self.sy,
                           0])

        pointC = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                           self.sy,
                           self.sz])
        
        # stack points to define the current state of the environment
        self.state = np.vstack([pointA, pointB, pointC])
    
    def render(self, state, with_seg=False, titleText=None, show=False):
        # get the slice and segmentation corresponding to the current state
        if with_seg:
            slice, seg = self.sample(state=state, return_seg=True) 
            slice, seg = slice.cpu().numpy().squeeze(), seg.cpu().numpy().squeeze()
            # stack image and segmentation, progate through channel dim since black and white
            # must do this to call ``ImageSequenceClip`` later.
            image = np.hstack([slice[..., np.newaxis] * np.ones(3), seg[..., np.newaxis] * np.ones(3)])
        else:
            slice = self.sample(state=state)
            image = slice[..., np.newaxis] * np.ones(3)

        # put title on image
        if titleText is not None:
            title = np.zeros((40, image.shape[1], image.shape[2]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # get boundary of this text
            textsize = cv2.getTextSize(titleText, font, 1, 2)[0]
            # get coords based on boundary
            textX = int((title.shape[1] - textsize[0]) / 2)
            textY = int((title.shape[0] + textsize[1]) / 2)
            # put text on the title image
            cv2.putText(title, titleText, (textX, textY ), font, 1, (255, 255, 255), 2)
            # stack title to image
            image = np.vstack([title, image])
        
        if show:
            # Show the image
            cv2.imshow("Environment", image)
            # This line is necessary to give time for the image to be rendered on the screen
            cv2.waitKey(1)
        else:
            return image


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

def get_plane_coefs(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d