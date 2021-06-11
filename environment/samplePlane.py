import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage, misc

def saveJPG(arr, path):
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path, quality=100)

class Point():
    def __init__(self, x, y, z, arr):
        self._x = x
        self._y = y
        self._z = z

        sx, sy, sz = arr.shape

        self.max_x = sx
        self.max_y = sy
        self.max_z = sz

    def __str__(self):
        return "Point ("+str(self._x)+","+str(self._y)+","+str(self._z)+")"

    def p_coords(self):
        return np.array([self._x, self._y, self._z])
    
    def coords(self):
        return np.array([self._x*self.max_x, self._y*self.max_y, self._z*self.max_z])
    
    def rot_ax(self, num=1):
        for _ in range(num):
            tmp = self._x
            self._x = self._y
            self._y = self._z
            self._z = tmp
        return self

    @property
    def x(self):
        return self._x*self.max_x
    
    @property
    def y(self):
        return self._y*self.max_y
    
    @property
    def z(self):
        return self._z*self.max_z

def draw_sphere(arr, point, r=3, color = 255):
    i,j,k = np.indices(arr.shape)
    dist = np.sqrt( (point.x-i)**2 + (point.y-j)**2 + (point.z-k)**2)
    arr[dist < r] = color
    return arr

def get_plane_coefs(pointA, pointB, pointC):
    p1 = pointA.coords()
    p2 = pointB.coords()
    p3 = pointC.coords()

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d

def get_plane(arr, coefs, oob_black=True):
    a, b, c, d = coefs
    sx, sy, sz = arr.shape
    main_ax = np.argmax([abs(a), abs(b), abs(c)])

    # print("main_ax =", main_ax+1)

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
    
    plane = arr[X, Y, Z]
    if oob_black == True:
        plane[P < 0] = 0
        plane[P > S] = 0

    return plane, (X, Y, Z)

def save_mpl(dotA, dotB, dotC, X, Y, Z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X.flatten(),
            Y.flatten(),
            Z.flatten(), 'bo ')

    ax.plot(*zip(dotA.p_coords(), dotB.p_coords(), dotC.p_coords()), color='r', linestyle=' ', marker='o')
    ax.view_init(0, 22)
    plt.tight_layout()
    plt.savefig('/vol/biomedic3/hjr119/XCAT/samples/planeMPL.jpg')

def sample_dot(limits):
    assert len(limits) == 3
    coords = []
    for ax in limits:
        if type(ax) == type(tuple()):
            v = (np.random.rand()*(max(ax)-min(ax))) + min(ax)
            coords.append(np.round(v, 3))
        else:
            coords.append(ax)
    
    return coords

def sample_planes(ct_arr, ranges, rep = 10):
    for i in range(rep):
        dotA        = Point(*sample_dot(ranges[0]), ct_arr)
        dotB        = Point(*sample_dot(ranges[1]), ct_arr)
        dotC        = Point(*sample_dot(ranges[2]), ct_arr)
        coefs       = get_plane_coefs(dotA, dotB, dotC)
        plane, _    = get_plane(ct_arr, coefs)
        angle       = np.round(np.random.rand()*10+40)
        plane       = ndimage.rotate(plane, angle, reshape=False, order=0)
        name        = "samp"+ ("0000"+str(i))[-4:]  + ".jpg"
        saveJPG(plane, os.path.join("/vol/biomedic3/hjr119/XCAT/samples/planes",name))

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

if __name__ == "__main__":

    ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/default_512_CT_1.nii.gz")
    ct_arr = sitk.GetArrayFromImage(ct_itk)
    # ct_arr = np.moveaxis(ct_arr, 0, -1)
    sx, sy, sz = ct_arr.shape
    ct_arr = ct_arr/ct_arr.max()*255
    ct_arr = intensity_scaling(ct_arr, pmin=150, pmax=200, nmin=0, nmax=255)
    print(ct_arr.min(), ct_arr.max(), ct_arr.shape)


    # dotA = Point(0, 0, .5, ct_arr)
    # dotB = Point(0, .5, 0, ct_arr)
    # dotC = Point(.5, 0, 0, ct_arr)

    # axial, coronal, sagittal
    rangeA = ((0.85, 1) , 0, (0.7,0.92))    #DONE
    rangeB = ((0.3,0.43), 1, 0) 
    rangeC = ((0.3,0.43), 1, 1)   

    # rangeA = ((0, 1), 0, (0, 1))
    # rangeB = ((0, 1), 1, (0, 1)) 
    # rangeC = ((0, 1), 1, (0, 1))   

    previ = time.time()

    sample_planes(ct_arr, (rangeA, rangeB, rangeC), rep=200)

    after = time.time()

    print("Generated in", after-previ, "sec.")

    exit()

    dotA = Point(*sample_dot(rangeA), ct_arr)
    dotB = Point(*sample_dot(rangeB), ct_arr)
    dotC = Point(*sample_dot(rangeC), ct_arr)

    print(dotA, dotB, dotC, sep='\n')

    if False:
        r = 5
        ct_arr = draw_sphere(ct_arr, dotA, r=r)
        ct_arr = draw_sphere(ct_arr, dotB, r=r)
        ct_arr = draw_sphere(ct_arr, dotC, r=r)

    coefs = get_plane_coefs(dotA, dotB, dotC)
    print(coefs)

    plane, (X, Y, Z) = get_plane(ct_arr, coefs)
    # print(coefs, plane.shape)
    print(plane.min(), plane.max())

    angle = np.round(np.random.rand()*10+40)
    print("Angle", angle)
    plane = ndimage.rotate(plane, angle, reshape=False)

    ct_arr[X, Y, Z] = 255 # Draws white lines on the images

    # name = "plane_"+str(hA)[:3]+"_"+str(hB)[:3]+"_"+str(hC)[:3]+".jpg"
    name = "tmp.jpg"
    saveJPG(plane, os.path.join("/vol/biomedic3/hjr119/XCAT/samples",name))

    saveJPG(ct_arr[sx//2, :, :], "/vol/biomedic3/hjr119/XCAT/samples/ax1.jpg")
    saveJPG(ct_arr[:, sy//2, :], "/vol/biomedic3/hjr119/XCAT/samples/ax2.jpg")
    saveJPG(ct_arr[:, :, sz//2], "/vol/biomedic3/hjr119/XCAT/samples/ax3.jpg")

    # save_mpl(dotA, dotB, dotC, X, Y, Z)

    print("done")