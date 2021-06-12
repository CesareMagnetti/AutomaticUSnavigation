# AutomaticUSnavigation

Investigating automatic navigation towards standard US views integrating RL/decision transformer with the virtual US environment developed in [CT2US simulation](https://github.com/CesareMagnetti/CT2UStransfer).

## example random walk trajectory
for now we have implemented the baseline plane sampling for the agent to move around the CT volume. The agent is in control of moving 3 points in a 3D volume, which will select the corresponding CT plane that is sampled. We model the agent to learn to move towards planes which maximize the visibility of a certain anatomical structure (i.e. the left ventricle) and we reward him according to the number of pixels in the 2D view that belong to the goal anatomical structure. Below is an example of a random walk through the CT volume (left) and the corresponding segmented slice (right). The reward at this time step will simply be the normalized count of pixels belonging to the left ventricle. The agent is incentivized to navigate towards a slice that mazimizes the visibility of the left ventricle.

<div align="center">
    <img width="70%" src="trajectories/sample.gif", alt="random walk through the CT volume and corresponding rewards."
	title="random walk through the CT volume and corresponding rewards." ><br>
	Fig 1: A random walk throug the CT volume with the corresponding (not normalized) reward. On the left we have the sampled CT planes by an agent following a 	    random walk, on the right we have the corresponding segmentation. The reward corresponds to the number of pixels in the 2D slice that belong to the left     	 ventricle.<br>
</div>
