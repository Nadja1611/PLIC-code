
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:23:55 2023

@author: Nadja Gruber and Martin Knoflach
"""

import torch
from unet_blocks import *
from fpdf import FPDF 
from read_in_DICOM import *
import numpy as np
from functions_test import *
from predict_test import *
import argparse
import SimpleITK as sitk
import logging
import sys





parser = argparse.ArgumentParser(description='Arguments for segmentation network.',add_help=False)
parser.add_argument('--outputdir', type=str, 
                    help='directory for outputs', default= "C://Users//nadja//Documents//PLIC_application_RC1_output")
parser.add_argument('--inputdir', type=str, 
                    help='directory for input files', default = "C://Users//nadja//Documents//Babies//" )
parser.add_argument('--logdir', type=str, 
                    help='directory for log files', default = "C://Users//nadja//Documents//PLIC_application_RC1_output//" )
#parser.add_argument('--weightdir', type=str, 
 #                   help='directory for weights', default = "C://Users//nadja//Documents//PLIC_application_RC1//" )
parser.add_argument('--Patient', type = int, help="which Patient do we want to be segmented", default = 97)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='This python application computes the volume of the posterior limb of the internal capsule of Babies on T1 MRI, and saves a txt file')
args = parser.parse_args()
logging.basicConfig(filename=args.logdir+'output.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# Create a custom logger
logger = logging.getLogger('output.log')

# Create a file handler for the logger
handler = logging.FileHandler(args.logdir + 'output.log')

# Add the handler to the logger
logger.addHandler(handler)

bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
print(bundle_dir)
#weight_path = os.path.abspath(os.path.join(bundle_dir, 'config.yml'))

script_dir = os.getcwd() 
weight_path = bundle_dir +"//"
# weight_path_ax = os.path.join(script_dir, 'best_dice_weights_T1.hdf5')
# weight_path_sag = os.path.join(script_dir, 'best_dice_weights_sag_T1.hdf5')
# weight_path_cor = os.path.join(script_dir, 'best_dice_weights_cor_T1.hdf5')

'''create the data '''
T1 = read_in(args.inputdir,args.Patient)[0]
T1_orig = read_in(args.inputdir,args.Patient)[5]
X_new = read_in(args.inputdir,args.Patient)[1]
indices = read_in(args.inputdir,args.Patient)[2]
Voxels = read_in(args.inputdir,args.Patient)[3]
Spacings = read_in(args.inputdir,args.Patient)[4]
header = read_in(args.inputdir,args.Patient)[6]

if header.StudyDescription == "Schädel":
    logging.info("The input has appropriate StudyDescription")
    X_new = np.expand_dims(X_new, axis=-1)
    X_new = np.moveaxis(X_new, 3,1)
    X_new = torch.tensor(X_new)
    
    data = torch.tensor(X_new)
    
    mynet = Segmentation_of_PLIC(inputs = "T1")
    mynet.data= data
    print(mynet.device)
    '''create the coronal and sagittal data '''
    mynet.create_sag_cor_patches_test( X_new, indices[0])
    mynet.standardise_coronal_sagittal(mynet.data_sag)
    mynet.standardise_coronal_sagittal(mynet.data_cor)
    '''compute weights for BCE '''
    mynet.init_NW(device=mynet.device) 
    logger.info(str(mynet.device) + " is used")
    ''' compute volume in mm^3 '''
    result=predict_and_compute_volume(mynet, weight_path, indices, Voxels, Spacings, T1_orig)
    volume = result[0]
    '''- predicted segmentation mask resized to original size '''
    segmented_image = result[1]
    ''' original T1 before resizing '''
    original_T1 = result[2]
    ''' generate dicom output '''
    #sitk_seg_image = sitk.GetImageFromArray(np.float32(segmented_image))
    sitk_seg_image = sitk.GetImageFromArray(np.uint8(segmented_image))

    #sitk_orig_T1 = sitk.GetImageFromArray(np.float32(original_T1))
    sitk_orig_T1 = sitk.GetImageFromArray((original_T1))
    sitk_orig_T1 = sitk.Cast(sitk_orig_T1, sitk.sitkUInt8)
    overlay_color_img = sitk.ScalarToRGBColormap(sitk_seg_image, 
                                             sitk.ScalarToRGBColormapImageFilter.Jet)
    
    # combined_volume = sitk.Cast(alpha_blend(sitk.Compose(sitk_orig_T1, sitk_orig_T1, sitk_orig_T1), 
    #                                     overlay_color_img), 
    #                         sitk.sitkVectorUInt8)
    red = [255,0,0]
    green = [0,255,0]
    blue = [0,0,255]
    combined_volume = sitk.LabelMapContourOverlay(sitk.Cast(sitk_seg_image, sitk.sitkLabelUInt8), 
                                                     sitk_orig_T1, 
                                                     opacity = 1, 
                                                     contourThickness=[1,1,1],
                                                    # dilationRadius= [3,3,3],
                                                     colormap=red)

    #original_T1[segmented_image==1]=255 
    
    header.SeriesDescription= "t1 mpr ax rek result"  
    #header.PatientName = "Nadja"
    #sitk_image = sitk.GetImageFromArray(original_T1)
    sitk_image = combined_volume
    vol = str(np.round(volume,3))
    sitk_image.SeriesDescription =  "t1 mpr ax rek result"     
    #sitk_image = sitk.GetImageFromArray(original_T1)
    sitk_image.SetOrigin(header.ImagePositionPatient)

    # Set DICOM series information
    series_tag_values = {
        "Modality": "MR",
        "SeriesDescription": "t1 mpr ax rek result"      
        # Add other necessary tags here
    }


    # Set metadata for the entire volume
    for tag, value in series_tag_values.items():
        sitk_image.SetMetaData(tag, value)
    # Get DICOM series writer
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Save the modified image as a DICOM series
    output_directory = args.outputdir
    os.makedirs(output_directory, exist_ok=True)


        # Save the volume
    output_filename = os.path.join(output_directory, f"segmentation.dcm")
    writer.SetFileName(output_filename)
    writer.Execute(sitk_image)

    logging.info("DICOM series saved successfully.")

    
    outputdir= args.outputdir
    for filename in os.listdir(outputdir):
        if filename.endswith(".dcm"):
            print(filename)
            dicom_filepath = os.path.join(outputdir, filename)
            dicom_instance = pydicom.dcmread(dicom_filepath)
        
        # Modify additional DICOM tags if needed
            dicom_instance.PatientName = str(header.PatientName)
            dicom_instance.PatientID = str(header.PatientID)
            dicom_instance.PatientAge = str(header.PatientAge)
            dicom_instance.StudyDate = str(header.StudyDate)
            dicom_instance.SeriesNumber = str(20000)

            dicom_instance.EchoTime = str(header.EchoTime)
            dicom_instance.RepetitionTime = str(header.RepetitionTime)
            dicom_instance.InstitutionName = str(header.InstitutionName)
            dicom_instance.Modality = str(header.Modality)
            dicom_instance.ReferringPhysicianName = str(header.ReferringPhysicianName)
            dicom_instance.OperatorsName = str(header.OperatorsName)
            dicom_instance.ImageComments = "The PLIC Volume is " + vol +" mm^3. NOT FOR CLINICAL USE!"
            #dicom_instance.ReferencedPerformedProcedureStepSequence = str(header.ReferencedPerformedProcedureStepSequence)
            dicom_instance.save_as(dicom_filepath)
    

    os.chdir(args.outputdir)
    lines = ["Computed PLIC volume",'Baby: '+ str(header.PatientID), 'Date: '+str(header.StudyDate),'Time: '+str(header.StudyTime), 'Age: '+ str(header.PatientAge),'Sex: '+ str(header.PatientSex), 'Description: '+header.StudyDescription, 'Volume:'+ str(np.round(volume,3)) + "mm^3"]
    with open('volume_'+str(header.PatientID)+'.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    
    pdf = FPDF()      
    # Add a page 
    pdf.add_page()  
    # set style and size of font  
    # that you want in the pdf 
    pdf.set_font("Arial", size = 10)
    # open the text file in read mode 
    f = open('volume_'+str(header.PatientID)+'.txt', "r") 
    # insert the texts in pdf 
    for x in f: 
        pdf.cell(50,5, txt = x, ln = 1, align = 'L') 
    # save the pdf with name .pdf 
    pdf.output('volume_'+str(header.PatientID)+".pdf")
else:
    logging.error("The input does not have appropriate StudyDescription")
    
   