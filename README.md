# Segmentation of posterior limb of internal capsule in preterm neonates

This repository contains the source code of the paper "A deep learning pipeline for the automated segmentation of posterior limb of internal
capsule in preterm neonates". In the following the steps of the framework are depicted.

<img src="./pipeline_diagram.png">

## Highlights
The main components of the proposed pipeline are as follows:

1. Extraction of a centred image patch and training of a classification network, the PLIC-Slice-Selector, on these patches to identify the slices containing the PLIC-voxels.
2. Training of a segmentation module, a U-Net, on the remaining slices to obtain axial probability masks of the PLIC.
3. Training of a classification network, the Thalamic-Slice-Selector, that identifies the slice that corresponds to the level of the boundary between upper and middle third of the thalamus.
4. The combination of the results from~\ref{2} and~\ref{3} yield ROI-boxes, from which the patches in coronal and sagittal plane view, arise.  
5. Training of a segmentation module on coronal and sagittal plane view patches. 
6. Combination of the resulting probability masks and creation of binary PLIC-masks.



## Training
