DATADIR=$1
CHECKPOINTDIR=$2
RESULTDIR=$3

# ========== GENRERATE TEST IMAGES AND RESULTS ==========

# ==== 1st experiment ====

# CycleGAN with standard configuration
python CycleGAN/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CycleGAN_standard --model cycle_gan --input_nc 1 --output_nc 1
# CUT with standard configuration
python CUT/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CUT_standard --model cut --CUT_mode CUT --input_nc 1 --output_nc 1

# ==== 2nd experiment ====

# CycleGAN without identity loss
python CycleGAN/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CycleGAN_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1
# CUT without identity loss
python CUT/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CUT_noIdtLoss --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1

# ==== 3rd experiment ====
# use LPIPS as a cycle consistency loss

# CycleGAN+LPIPS with the idt loss
python CycleGAN/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CycleGAN_LPIPS --model cycle_gan --input_nc 1 --output_nc 1
# CycleGAN+LPIPS without the idt loss
python CycleGAN/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CycleGAN_LPIPS_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1
# # CycleGAN+LPIPS without the idt loss and using a scaling factor of 1 (instead of 10) for the LPIPS loss
python CycleGAN/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CycleGAN_LPIPS_noIdtLoss_lambda_AB_1 --model cycle_gan --input_nc 1 --output_nc 1

# ==== 4th experiment ====

# CUT with standard configuration and removed layer 0 from PatchNCE loss
python CUT/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CUT_standard_noLayer0 --model cut --CUT_mode CUT --input_nc 1 --output_nc 1
# CUT without identity loss and removed layer 0 from PatchNCE loss
python CUT/test.py --dataroot $DATADIR --checkpoints_dir $CHECKPOINTDIR --results_dir $RESULTDIR --name CUT_noIdtLoss_noLayer0 --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1

# ========== CALCULATE FID ==========

# cycle_gan test.py does not store files in the respective folders but stores them as <*_real_A.png> or <*_real_B.png> etc.
# pythorch_fid expects files in folders. CUT test.py already stores files in folders.
bash rearrange_files_cycle_gan.sh ${RESULTDIR}CycleGAN_standard/test_latest/images/
bash rearrange_files_cycle_gan.sh ${RESULTDIR}CycleGAN_noIdtLoss/test_latest/images/
bash rearrange_files_cycle_gan.sh ${RESULTDIR}CycleGAN_LPIPS/test_latest/images/
bash rearrange_files_cycle_gan.sh ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss/test_latest/images/
bash rearrange_files_cycle_gan.sh ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss_lambda_AB_1/test_latest/images/

# calculate FID for each model
echo -e "CycleGAN_standard:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CycleGAN_standard/test_latest/images/real_B ${RESULTDIR}CycleGAN_standard/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCUT_standard:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CUT_standard/test_latest/images/real_B ${RESULTDIR}CUT_standard/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCycleGAN_noIdtLoss:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CycleGAN_noIdtLoss/test_latest/images/real_B ${RESULTDIR}CycleGAN_noIdtLoss/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCUT_noIdtLoss:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CUT_noIdtLoss/test_latest/images/real_B ${RESULTDIR}CUT_noIdtLoss/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCycleGAN_LPIPS:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CycleGAN_LPIPS/test_latest/images/real_B ${RESULTDIR}CycleGAN_LPIPS/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCycleGAN_LPIPS_noIdtLoss:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss/test_latest/images/real_B ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCycleGAN_LPIPS_noIdtLoss_lambda_AB_1:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss_lambda_AB_1/test_latest/images/real_B ${RESULTDIR}CycleGAN_LPIPS_noIdtLoss_lambda_AB_1/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCUT_standard_noLayer0:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CUT_standard_noLayer0/test_latest/images/real_B ${RESULTDIR}CUT_standard_noLayer0/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt
echo -e "\nCUT_noIdtLoss_noLayer0:\t" >> ${RESULTDIR}FID_scores.txt
python -m pytorch_fid ${RESULTDIR}CUT_noIdtLoss_noLayer0/test_latest/images/real_B ${RESULTDIR}CUT_noIdtLoss_noLayer0/test_latest/images/fake_B >> ${RESULTDIR}FID_scores.txt