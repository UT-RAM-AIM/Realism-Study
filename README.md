# Computer-Assisted-Learning
Repository contains code for Computer Assisted Learning; data preprocessing, obtaining synthetic images and realism assessment.

Information for the manuscript: "Perceptual Study of Semantically Synthesized 2D Chest CT Image Realism: Quantitative Metrics and Expert Opinion". 

## Data preprocessing
The publicly available dataset of [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) is used for this research and should be downloaded via their website. The script 'preprocessing_CT.py' will convert the DICOM files into 2D original images, with corresponding semantic label map and visual label map. They will be stored in the folders 'image', 'label', and 'put' respectively. 

The pre-processed data can be divided into a combined training and validation set, and a test set. In the manuscript, subjects 1 through 900 are used for training validation and 901 through 1010 for testing. Additionally, for the experiments with a smaller training set, the first 2% of the full training set are used. In case of 0.3%, only subject 1 is used (without 20% sampling).

## Obtaining synthetic images
As semantic image synthesis network, the model developed by Park et al. is used, referred to as [SPADE](https://github.com/NVlabs/SPADE) or sometimes as GauGAN. Their code requires the semantic label maps from 'label' as input, as well as the images from 'image' as guiding image. This is similar in both the training and testing phase. During testing the created synthetic images will be saved in an output directory.

## Realism assessment
To obtain quantitative information about the realism of the synthetic images, the 'quantitative__metrics.py' script can be used. It requires the paths to the location of the synthetic images and the original images. The preferred metric can be indicated; psnr, ssim, ms-ssim and lpips. The FID can be obtained via the code from SPADE.

In the script 'qualitative_perceptual_study.py', code is provided to set-up the perceptual study, i.e. create 60 quartets of images and save information about their origin.
