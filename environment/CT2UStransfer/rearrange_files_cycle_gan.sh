# re-arrange images for cycle_gan models, CUT already stores them in folders
MYPATH=$1
mkdir -p ${MYPATH}real_A ${MYPATH}real_B ${MYPATH}fake_A ${MYPATH}fake_B ${MYPATH}rec_A ${MYPATH}rec_B
mv ${MYPATH}*real_A.png ${MYPATH}real_A
mv ${MYPATH}*real_B.png ${MYPATH}real_B
mv ${MYPATH}*fake_A.png ${MYPATH}fake_A
mv ${MYPATH}*fake_B.png ${MYPATH}fake_B
mv ${MYPATH}*rec_A.png ${MYPATH}rec_A
mv ${MYPATH}*rec_B.png ${MYPATH}rec_B