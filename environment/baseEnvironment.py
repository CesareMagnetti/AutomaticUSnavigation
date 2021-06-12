import numpy as np
import SimpleITK as sitk
import os

class BaseEnvironment():
    def __init__(self, dataroot, volume_id, segmentation_id, start_pos = None, scale_intensity=True, rewardID=2885, **kwargs):
        
        # load CT volume
        itkVolume = sitk.ReadImage(os.path.join(dataroot, volume_id+".nii.gz"))
        Volume = sitk.GetArrayFromImage(itkVolume)
        self.sx, self.sy, self.sz = Volume.shape

        # load CT segmentation
        itkSegmentation = sitk.ReadImage(os.path.join(dataroot, segmentation_id+".nii.gz"))
        Segmentation = sitk.GetArrayFromImage(itkSegmentation)

        # preprocess volume
        Volume = Volume/Volume.max()*255
        if scale_intensity:
            Volume = intensity_scaling(Volume, pmin=kwargs.pop('pmin', 150), pmax=kwargs.pop('pmax', 200),
                                       nmin=kwargs.pop('nmin', 0), nmax=kwargs.pop('nmax', 255))
        
        # save as class attribute
        self.Volume = Volume
        self.Segmentation = Segmentation

        # ID of the segmentation corresponding to the anatomical structure we are trying to observe.
        self.rewardID = rewardID

        # get starting configuration
        if start_pos:
            self.pointA = start_pos.pointA
            self.pointB = start_pos.pointB
            self.pointC = start_pos.pointC
        else:
            # get a meaningful starting plane (these hardcoded ranges will yield views close to 4-chamber view) 
            self.pointA = np.array([np.random.uniform(low=0.85, high=1)*self.sx,
                                    0,
                                    np.random.uniform(low=0.7, high=0.92)*self.sz])
            self.pointB = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                                    self.sy,
                                    0])
            self.pointC = np.array([np.random.uniform(low=0.3, high=0.43)*self.sx,
                                    self.sy,
                                    self.sz])


    def sample(self, oob_black=True):

        # get plane coefs
        a,b,c,d = get_plane_coefs(self.pointA, self.pointB, self.pointC)

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
        
        segmentation = self.Segmentation[X, Y, Z]

        # sample the according reward (i.e. count of pixels in left ventricle)
        # the left ventricle ID in the segmentation is 2885. let's count the number
        # of pixels in the left ventricle as a fit function, the agent will have to
        # find a view that maximizes the amount of left ventricle present in the image.
        unique, counts = np.unique(segmentation, return_counts=True)
        reward = counts[unique==self.rewardID]

        return plane, reward, segmentation

    def step(self, dirA, dirB, dirC):

        # update current position
        self.pointA += dirA
        self.pointB += dirB
        self.pointC += dirC

        # sample the plane, reward and the segmentation map
        nextState, reward, segmentation = self.sample()

        return nextState, reward, segmentation

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