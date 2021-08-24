"""Script adapted from NST torch tutorial to translate XCAT volumes to absorb CT texture
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import concurrent.futures
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.detach().clone().view(-1, 1, 1)
        self.std = std.detach().clone().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=10000, content_weight=1):
    """Run the style transfer."""

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            # if run[0] % 50 == 0:
            #     print("run {}:".format(run))
            #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            #         style_score.item(), content_score.item()))
            #     print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


# load and preprocess an XCAT volume
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

def load(filename, pmin, pmax, nmin, nmax):
    itkVolume = sitk.ReadImage(filename)
    Volume = sitk.GetArrayFromImage(itkVolume) 
    # preprocess volume
    Volume = Volume/Volume.max()*255
    Volume = intensity_scaling(Volume, pmin=pmin, pmax=pmax, nmin=nmin, nmax=nmax)
    Volume = Volume.astype(np.uint8)
    return Volume

# ==== handle style transfer for each slice of a volume ====
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

totensor = transforms.ToTensor()
resize = transforms.Resize(256)
noise = AddGaussianNoise(mean=0, std=0.5)
noise1 = AddGaussianNoise(mean=0, std=0.1)
def NST_Volume(Volume, style_img, main_ax, init="content"):
    # 1. roll volume axis as queried (to slice volume along x,y or z)
    np.rollaxis(Volume, main_ax)
    print(Volume.shape)
    sx,sy,sz = Volume.shape
    # 2. loop through all slices along main_Axis, transfer the style and update slices in the Volume
    for i in tqdm(range(100, sx), "transfering volume ..."):
        # 2.1 setup content and input images to tensor already sets image in (0,1) range
        content_img = noise1(totensor(Volume[i, ...])).unsqueeze(0).to(device, torch.float)
        if init == "content":
            input_img = noise(totensor(Volume[i, ...])).unsqueeze(0).to(device, torch.float)
        else:
            input_img = torch.randn(content_img.data.size(), device=device)
        # 2.2 run style transfer on this plane
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
        output = output.cpu().detach().numpy().squeeze()
        img = content_img.cpu().detach().numpy().squeeze()
        img = np.hstack([img, output])
        # save as we go to inspect (DELETE AFTER)
        if not os.path.exists("./temp_transferred_slices"):
            os.makedirs("./temp_transferred_slices")
        plt.imsave("./temp_transferred_slices/sample{}.png".format(i), img, cmap="Greys_r")
        # 2.3 store output image in the volume
        Volume[i, ...] = output

    # 3. roll the axis back to its origin form
    np.rollaxis(Volume, 0, main_ax)
    return Volume



parser = argparse.ArgumentParser(description='train/test scripts to launch navigation experiments.')
parser.add_argument('--dataroot', '-r',  type=str, default="/vol/biomedic3/hjr119/XCAT/generation2", help='path to the XCAT CT volumes.')
parser.add_argument('--saveroot', '-s',  type=str, default="./XCAT_VOLUMES_NST/", help='path to the XCAT CT volumes.')
parser.add_argument('--volume_ids', '-vol_ids', type=str, default='samp0', help='filename(s) of the CT volume(s) comma separated.')
parser.add_argument('--style_img', '-si', type=str, default='./LIDC-IDRI-0139_70.jpg', help='path to style image.')
parser.add_argument('--main_ax', type=int, default=0, help="main_ax to follow when translating each slice of the volume.")
parser.add_argument('--init', '-i',  type=str, default="content", help='initialize input image either to [content] or [random] image.')
parser.add_argument('--pmin', type=int, default=150, help="pmin value for intensity_scaling() function.")
parser.add_argument('--pmax', type=int, default=200, help="pmax value for intensity_scaling() function.")
parser.add_argument('--nmin', type=int, default=0, help="nmin value for intensity_scaling() function.")
parser.add_argument('--nmax', type=int, default=255, help="nmax value for intensity_scaling() function.")
config = parser.parse_args()

if __name__ == "__main__":
    vol_ids = config.volume_ids.split(",")
    # main function to transfer a volume
    def main(vol_id):
        # 1. load and preprocess the XCAT volume
        Volume = load(os.path.join(config.dataroot, vol_id+"_1_CT.nii.gz"), config.pmin, config.pmax, config.nmin, config.nmax)
        # 2. load and preprocess the style CT image
        style_img_numpy = Image.open(config.style_img).convert('L')
        style_img = resize(totensor(style_img_numpy)).unsqueeze(0).to(device, torch.float)
        # 3. transfer the volume
        TransferredVolume = NST_Volume(Volume, style_img, config.main_ax, init=config.init)
        # 4. save volume
        # ======== THESE PARAMS ARE TAKEN FROM vol/biomedic3/hjr119/XCAT/gen_xcat.py =======
        ZOOM = 3
        BASE = 256/ZOOM # Default
        CM_TO_PIX = 0.3125*BASE
        final_dim = 256
        dim_2_percentage_stop  = .65 # haut bas
        dim_2_percentage_start = 0. # 0. - 0.75
        plane_dim = int(np.ceil(final_dim / (dim_2_percentage_stop-dim_2_percentage_start)))
        pix_res = CM_TO_PIX/plane_dim
        # =======================================================================================
        # Transforming numpy to sitk
        print("Transforming numpy to sitk...")
        sitk_arr = sitk.GetImageFromArray(TransferredVolume)
        sitk_arr.SetSpacing([pix_res, pix_res, pix_res])

        # Save sitk to nii.gz file
        print("Saving sitk to .nii.gz file...", )
        writer = sitk.ImageFileWriter()
        if not os.path.exists(config.saveroot):
            os.makedirs(config.saveroot)
        writer.SetFileName(os.path.join(config.saveroot, vol_id+"_1_CT.nii.gz"))
        writer.Execute(sitk_arr)

    # parallelize across all input volumes
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(main, vol_id) for vol_id in vol_ids]


