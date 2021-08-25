import SimpleITK as sitk
import numpy as np
import os
import argparse
import scipy.signal

parser = argparse.ArgumentParser(description='smooth fakeCT volumes.')
parser.add_argument('--dataroot', '-r',  type=str, default="./XCAT_VOLUMES_NST/", help='path to the fake CT volumes.')
parser.add_argument('--volume_ids', '-vol_ids', type=str, default='samp0', help='filename(s) of the CT volume(s) comma separated.')
args = parser.parse_args()

if __name__ == "__main__":
    vol_ids = args.volume_ids.split(",")
    filenames_volumes = [os.path.join(args.dataroot, vol_id+"_1_CT.nii.gz") for vol_id in vol_ids]
    filenames_segmentations = [os.path.join(args.dataroot, vol_id+"_1_SEG.nii.gz") for vol_id in vol_ids]

    for i in range(len(vol_ids)):
        # load CT volume
        itkVolume = sitk.ReadImage(filenames_volumes[i])
        Spacing = itkVolume.GetSpacing()
        Volume = sitk.GetArrayFromImage(itkVolume) 

        # smoothen CT volume
        #Volume_smooth3 = scipy.signal.savgol_filter(Volume, window_length=9, polyorder=1, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
        Volume_smooth3 = scipy.signal.savgol_filter(Volume, window_length=5, polyorder=1, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
        # normalize smoothened volume
        Volume_smooth3 = ((Volume_smooth3-Volume_smooth3.min())/(Volume_smooth3.max()-Volume_smooth3.min())*255).astype(np.uint8)

        # 4. save the smoothened volume
        # Transforming numpy to sitk
        print("Transforming numpy to sitk...")
        sitk_arr = sitk.GetImageFromArray(Volume_smooth3)
        sitk_arr.SetSpacing(Spacing)

        # Save sitk to nii.gz file
        print("Saving sitk to .nii.gz file...", )
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(args.dataroot, vol_ids[i]+"_smooth_1_CT.nii.gz"))
        writer.Execute(sitk_arr)