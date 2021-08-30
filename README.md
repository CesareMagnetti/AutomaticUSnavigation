# AutomaticUSnavigation

Investigating automatic navigation towards standard US views integrating MARL with the virtual US environment developed in [CT2US simulation](https://github.com/CesareMagnetti/CT2UStransfer). We will start by investigating navigation in the XCAT phantom volumes, then integrate our cycleGAN model to the pipeline to perform navigation in US domain. We also test navigation on clinical CT scans.

## example of agents navigating in a test XCAT phantom volume (not seen at train time)

The agent is in control of moving 3 points in a 3D volume, which will sample the corresponding plane. We aim to model the agent to learn to move towards 4-chamber views. We define such views as the plane passing through the centroids of the Left Ventricle, Right Ventricle and Right Atrium (XCAT volumes come with semantic segmentations). We reward the agent when it moves towards this goal plane, and when the number of pixels of tissues of interest present in the current plane increase (see rewards/rewards.py fro more details). Furthermore, we add some good-behaviour inducing reards: we maximize the area of the triangle spanned by the agents and we penalize the agents for moving outside of the volumes boundaries. The former encourages smooth transitions (if the agents are clustered close together we would get abrupt transitions) the latter makes sure that the agents stay within the boundaries of the environment. The following animation shows agents navigating towards a 4-Chamber view on a test XCAT volume, agents are initialized randomly within the volume.

<div align="center">
    <img width="40%" src="readme_images/standardXCATfullTrajectory.gif", alt="trained agent acting greedily."
	title="untrained agent acting greedily." ><br>
	Fig 1: Our best agent acting greedily for 250 steps after random initialization. Our full agent consists of 3 sub-agents, each controlling the movement of 1 	     point in a 3D space. As each agent moves around the 3 points will sample a particular view of the CT volume.<br>
</div>

## example of agents navigating in clinical CTs
We than upgrade our pipeline generating realistic fake CT volumes using Neural Style Transfer on our XCAT volumes. We will generate volumes which aim to resemble CT texture while retaining XCAT content. We train the agents in the same manner on this new simulated environment and we test practicality both on unseen fake CT volumes and on clinical volumes from LIDC-IDRI dataset. 

<div align="center">
    <img width="40%" src="readme_images/standardXCATfullTrajectory.gif", alt="trained agent acting greedily on fake CT."
	title="untrained agent acting greedily." ><br>
    <img width="40%" src="readme_images/standardXCATfullTrajectory.gif", alt="trained agent acting greedily on real CT."
	title="untrained agent acting greedily." ><br>
	Fig 2: Left) Our best agent acting greedily on a test fake CT volume for 125 steps after random initialization. Right) same agents tested on clinical CT data.<br>
</div>
## usage

1. clone the repo and install dependencies

```bash
git clone git@github.com:CesareMagnetti/AutomaticUSnavigation.git
cd AutomaticUSnavigation
python3 -m venv env
source env/bin/activate
pip install -r requirements
```

2. if you don't want to integrate the script with weights and biases run scripts with the additional ```--wandb disabled``` flag.

3. run our default untrained agent on the default volume. It will save a 250 steps animation to ```./results/untrained_agent/test/sample_0.gif```.

```bash
python test_visual.py --name untrained_agent --n_runs 1 --n_steps 250
```

3. train our default agent on the default volume to navigate towards a 2D view that maximizes the number of pixels in the Left Ventricle (or any other anatomical structure). In this case we are training for 2000 episodes of 250 steps each (takes approximately 3 hours on a GTX TITANX NVIDIA 12GB).

```bash
python train.py -r --name default_agent --n_episodes 2000 --n_steps_per_episode 250
```


## Acknowledgements
Work done with the help of [Hadrien Reynaud](https://github.com/HReynaud). Our CT2US models are built upon the [CT2US simulation](https://github.com/CesareMagnetti/CT2UStransfer) repo, which itself is heavily based on [CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation) repos.


