# Segmentation of posterior limb of internal capsule in preterm neonates

This repository contains the source code of the paper "A deep learning pipeline for the automated segmentation of posterior limb of internal
capsule in preterm neonates". In the following the steps of the framework are depicted.

<img src="./pipeline_diagram.png">

## Highlights
The main components of the proposed pipeline are as follows:

1. Extraction of a centred image patch and training of a classification network, the PLIC-Slice-Selector, on these patches to identify the slices containing the PLIC-voxels.
2. Training of a segmentation module, a U-Net, on the remaining slices to obtain axial probability masks of the PLIC.
3. Training of a classification network, the Thalamic-Slice-Selector, that identifies the slice that corresponds to the level of the boundary between upper and middle third of the thalamus.
4. The combination of the results from 2. and 3. yield ROI-boxes, from which the patches in coronal and sagittal plane view, arise.  
5. Training of a segmentation module on coronal and sagittal plane view patches. 
6. Combination of the resulting probability masks and creation of binary PLIC-masks.



## Training
1. conda create --name <env> --file requirements.txt
2. In functions.py the functions used for preprocessing and postprocessing, as well as for evaluation can be found
3. In models.py the slice selection modules as well as the segmentations modules can be found
4. In the folder "train" these models are trained, weights and output data are generated. 
 

   The order of execution of the code is the following:
   * Plic-Slice-Selector.py
   * Thalamic-Slice-Selector.py
   * axial_segmentation_modle.py
   * coronal_segmentation_module.py, sagittal_segmentation_module.py
   * combination_and_plot.py
 PLIC-slice selector prepares the ouputs required for further steps. It contains the patches containing PLIC, their indices, as well as the indices of the Thalamic-level slices only restricted on the ROI. These indices are then further used to create sagittal and coronal patches with the center slice being the one between middle and upper third of the thalamus. 
 
 ## Data
 The data needed for executing this code consists of a file "Training_Babies.npz" containing the MRI images and the hand annotated segmentation masks and the number of slices per volume, as well as a file "Labels_Thal.npz" containing the slice indices of the slice at the mid-thalamic level in the brain. 
