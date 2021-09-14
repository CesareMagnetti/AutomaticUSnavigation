import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image
from numpy.random import default_rng

############ START OF PARAMS ############
# Generation time:
# 128 =          ,31s for 2
# 256 = 47s for 1, 67s for 2
# 512 = 10 minutes
# 1024 = 30 minutes

final_dim = 512
OUTPUT_TYPES = ['CT', 'SEG'] # ['CT', 'SEG', 'NURBS', 'ATN']
name = sys.argv[1] if len(sys.argv) > 1 else "samp"

type2pht = {'CT':'_atn_', 'SEG':'_act_', 'NURBS':'_'}
type2ext = {'CT':'.bin', 'SEG':'.bin', 'NURBS':'.nrb'}
# gen_path = "/vol/biomedic3/hjr119/XCAT/generation128/"
gen_path = "/vol/biomedic3/hjr119/XCAT/generation512"

os.makedirs(gen_path, exist_ok=True)

use_random_sensible = True         # If set to true, all values will be randomized, while staying in a sensible range
generate = True
clean_bin = True
SEG_LV_ONLY = False


slice_precentage_start = 1.-.2 # 0, 1-.18
slice_precentage_stop  = 1.-.08 # 1 - 0.11
# WARNING: These depends on the ZOOM factor
# which is defined below
ZOOM = 2 #3.5
# # IF ROTATED:
# dim_2_percentage_start = 0. # 0. - 0.75
# dim_2_percentage_stop  = .65 # haut bas
# dim_3_percentage_start = 0.10#.35 # 0.2 - 0.95
# dim_3_percentage_stop  = 0.75#1. # Gauche droite

# IF NOT ROTATED:
# dim_2_percentage_start = .0 # 0. - 0.75
# dim_2_percentage_stop  = .7 # haut bas
# dim_3_percentage_start = .2#.35 # 0.2 - 0.95
# dim_3_percentage_stop  = .9#1. # Gauche droite

slice_precentage_start = 0 # 0, 1-.18
slice_precentage_stop  = 1 # 1 - 0.11
dim_2_percentage_start = .0 # 0. - 0.75
dim_2_percentage_stop  = 1. # haut bas
dim_3_percentage_start = .0#.35 # 0.2 - 0.95
dim_3_percentage_stop  = 1.#1. # Gauche droite


# Heart adjustements
apical_thin = 0.0	                # apical_thinning (0 to 1.0 scale, 0.0 = not present, 0.5 = halfway present, 1.0 = completely thin)
uniform_heart = 0	                # sets the thickness of the LV (0 = default, nonuniform wall thickness; 1 = uniform wall thickness for LV)
hrt_start_ph_index = 0.3	        # hrt_start_phase_index (range=0 to 1; ED=0, ES=0.4) see NOTE 3 
heart_base = "vmale50_heart.nrb"	# basename for heart files (male = vmale50_heart.nrb; female = vfemale50_heart.nrb)
lv_radius_scale = 1		            # lv_radius_scale (value from 0 to 1 to scale the radius of the left ventricle)
lv_length_scale = 1		            # lv_length_scale (value from 0 to 1 to scale the length of the left ventricle)
hrt_scale_x = 1.0		            # hrt_scale x  
hrt_scale_y = 1.0		            # hrt_scale y  
hrt_scale_z = 1.0		            # hrt_scale z  
motion_defect_flag = 0	            # (0 = do not include, 1 = include) regional motion abnormality in the LV as defined by heart lesion parameters see NOTE 9

bones_scale = 1.0	                # bones_scale (scales all bones in 2D about their centerlines, makes each bone thicker)  SEE NOTE 5
thickness_ribs = 0.3		        # thickness ribs     (cm)
thickness_backbone = 0.4	        # thickness backbone (cm)
valve_thickness = 0.1		        # thickness of the AV valves (cm); (def is 0.1)
hrt_v1 = 0.0		                # sets the LV end-diastolic volume (0 = do not change); see NOTE 3A
hrt_v2 = 0.0		                # sets the LV end-systolic volume (0 = do not change); see NOTE 3A
hrt_v3 = 0.0		                # sets the LV volume at the beginning of the quiet phase (0 = do not change); see NOTE 3A
hrt_v4 = 0.0		                # sets the LV volume at the end of the quiet phase (0 = do not change); see NOTE 3A
hrt_v5 = 0.0		                # sets the LV volume during reduced filling, before end-diastole (0 = do not change); see NOTE 3A

# Global rotation
phan_rotx = 0.0	# degree to rotate the entire phantom by the x-axis
phan_roty = 0.0	# degree to rotate the entire phantom by the y-axis
phan_rotz = 0.0	# degree to rotate the entire phantom by the z-axis

#Generate a sequence
out_period = 1		                # output_period (SECS) (if <= 0, then output_period=time_per_frame*output_frames)
out_frames = 1		                # output_frames (# of output time frames )

############## END OF PARAMS ##############



################################################
##### DO NOT MODIFY PARAMS BELOW THIS LINE #####
################################################
    
plane_dim = int(np.ceil(final_dim / (dim_2_percentage_stop-dim_2_percentage_start)))

path = "/vol/biomedic3/hjr119/XCAT/program"
os.chdir(path)
BASE = 256/ZOOM # Default
CM_TO_PIX = 0.3125*BASE 
SLICE_MIN = 1
SLICE_MAX = 500*(plane_dim/BASE)
pix_res = CM_TO_PIX/plane_dim
slice_bot = int(max(SLICE_MIN, np.floor(SLICE_MAX*slice_precentage_start)))
slice_top = int(min(SLICE_MAX, np.floor(SLICE_MAX*slice_precentage_stop)))

if use_random_sensible:
    rng = default_rng() # No fixed seed for multi threading

    apical_thin         = rng.random() / 5 + 0.9 # np.round(rng.random()*1.1, 1) 
    # uniform_heart       = 1 if rng.random() > 0.9 else 0 # 10% chance	                
    hrt_start_ph_index  = np.round(rng.random()*1.1, 1)
    heart_base          = rng.choice(["vmale50_heart.nrb", "vfemale50_heart.nrb"])
    lv_radius_scale     = rng.random() / 5 + 0.9 # min(1.0, rng.lognormal(0.05, 0.3, None))
    lv_length_scale     = rng.random() / 5 + 0.9 # min(1.0, rng.lognormal(0.05, 0.3, None))
    hrt_scale_x         = rng.random() / 5 + 0.9 # min(1.0, rng.lognormal(0.05, 0.3, None)) 
    hrt_scale_y         = rng.random() / 5 + 0.9 # min(1.0, rng.lognormal(0.05, 0.3, None))
    hrt_scale_z         = rng.random() / 5 + 0.9 # min(1.0, rng.lognormal(0.05, 0.3, None))
    # motion_defect_flag  = 1 if rng.random() > 0.9 else 0 # 10% chance	 
    bones_scale         = 1.0 * (0.8 + rng.random()*0.4)
    thickness_ribs      = 0.3 * (0.9 + rng.random()*0.2)
    thickness_backbone  = 0.4 * (0.9 + rng.random()*0.2)
    valve_thickness     = 0.1 * (0.8 + rng.random()*0.4)

    hrt_v1 = 155.0  * (0.7 + rng.random()*0.4)
    hrt_v2 = 60.0   * (0.7 + rng.random()*0.4)   
    hrt_v3 = 95.0   * (0.7 + rng.random()*0.4)
    hrt_v4 = 130.0  * (0.7 + rng.random()*0.4)
    hrt_v5 = 145.0  * (0.7 + rng.random()*0.4)
    
get_act = 1 if 'SEG' in OUTPUT_TYPES else 0
get_seg = 1 if 'SEG' in OUTPUT_TYPES else 0
get_atn = 1 if 'CT' in OUTPUT_TYPES else 0
get_nrb = 1 if 'NURBS' in OUTPUT_TYPES else 0

# Phantom generation
start = time.time()
bashCmd =   "./dxcat2_linux_64bit general.samp.par "            + \
            "--startslice "         + str(slice_bot)            +" "+ \
            "--endslice "           + str(slice_top)            +" "+ \
            "--pixel_width "        + str(pix_res)              +" "+ \
            "--slice_width "        + str(pix_res)              +" "+ \
            "--array_size "         + str(plane_dim)            +" "+ \
            "--act_phan_each "      + str(get_act)              +" "+ \
            "--out_frames "         + str(out_frames)           +" "+ \
            "--color_code "         + str(get_seg)              +" "+ \
            "--atten_phan_each "    + str(get_atn)              +" "+ \
            "--apical_thin "        + str(apical_thin)          +" "+ \
            "--uniform_heart "      + str(uniform_heart)        +" "+ \
            "--hrt_start_ph_index " + str(hrt_start_ph_index)   +" "+ \
            "--heart_base "         + str(heart_base)           +" "+ \
            "--lv_radius_scale "    + str(lv_radius_scale)      +" "+ \
            "--lv_length_scale "    + str(lv_length_scale)      +" "+ \
            "--hrt_scale_x "        + str(hrt_scale_x)          +" "+ \
            "--hrt_scale_y "        + str(hrt_scale_y)          +" "+ \
            "--hrt_scale_z "        + str(hrt_scale_z)          +" "+ \
            "--motion_defect_flag " + str(motion_defect_flag)   +" "+ \
            "--nurbs_save "         + str(get_nrb)              +" "+ \
            "--motion_option "      + str(2)                    +" "+ \
            "--bones_scale "        + str(bones_scale)          +" "+ \
            "--thickness_ribs "     + str(thickness_ribs)       +" "+ \
            "--thickness_backbone " + str(thickness_backbone)   +" "+ \
            "--valve_thickness "    + str(valve_thickness)      +" "+ \
            "--hrt_v1 "             + str(hrt_v1)               +" "+ \
            "--hrt_v2 "             + str(hrt_v2)               +" "+ \
            "--hrt_v3 "             + str(hrt_v3)               +" "+ \
            "--hrt_v4 "             + str(hrt_v4)               +" "+ \
            "--hrt_v5 "             + str(hrt_v5)               +" "+ \
            "--phan_rotx "          + str(phan_rotx)            +" "+ \
            "--phan_roty "          + str(phan_roty)            +" "+ \
            "--phan_rotz "          + str(phan_rotz)            +" "+ \
            os.path.join(gen_path, name)#"../generation/"+name

print(bashCmd)
output = '\\n'*10
if generate:
    process = subprocess.Popen(bashCmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
stop = time.time()
print("Generated in ", stop-start, "seconds.")
# print("Generated slice ",slice," in ", stop-start, "s")
# print(str(output).split('\\n')[-3])

for i in range(1, out_frames+1):
    for output_type in OUTPUT_TYPES:

        canvas = None

        # Openning binary file
        print("Loading binary",i,"of",output_type)
        iname = name+type2pht[output_type]+str(i)+type2ext[output_type]
        oname = name+"_"+str(i)+"_"+output_type+".nii.gz"
        f=open(os.path.join(gen_path, iname), "rb")
        content = f.read()

        # Transforming binary to numpy array
        print("Transforming binary to numpy...")
        arr = np.frombuffer(content, dtype=np.float32)
        arr = arr.reshape(slice_top-slice_bot+1, plane_dim, plane_dim)
        arr = np.flip(arr, axis=0) # put feets at the bottom

        # Delete content to save ram
        content = None

        # Adjusting array dimensions to focus on the heart
        sx, sy, sz = arr.shape
        print("Original size is", arr.shape)
        d2s, d2e, d3s, d3e = int(dim_2_percentage_start*sy), int(dim_2_percentage_stop*sy), int(dim_3_percentage_start*sz), int(dim_3_percentage_stop*sz)
        if d2e-d2s != final_dim:
            d2e += final_dim-(d2e-d2s)
        if d3e-d3s != final_dim:
            d3e += final_dim-(d3e-d3s)
        arr = arr[:(d2e-d2s), d2s:d2e, d3s:d3e]
        arr = arr[::-1,:,:]
        print("Final array size is", arr.shape, "with pix res:", pix_res)

        # Transforming numpy to sitk
        print("Transforming numpy to sitk...")
        sitk_arr = sitk.GetImageFromArray(arr)
        sitk_arr.SetSpacing([pix_res, pix_res, pix_res])

        # Save sitk to nii.gz file
        print("Saving sitk to .nii.gz file...", )
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(gen_path, oname))
        writer.Execute(sitk_arr)
        if clean_bin:
            if os.path.exists(os.path.join(gen_path, iname)):
                os.remove(os.path.join(gen_path, iname))

        # Generate sample
        sx, sy, sz = arr.shape
        middle_cut = arr[int((1.5*sx)//2),:,:]
        middle_cut = middle_cut/middle_cut.max()*255

        if type(canvas) == type(None):
            canvas = middle_cut
        else:
            canvas = np.concatenate((canvas, middle_cut), axis=1)

        Image.fromarray(canvas.astype(np.uint8)).save(os.path.join(gen_path, name+"_"+output_type+".jpg"), quality=100)
        print()





