# Computer-Assisted-Learning
Information for the manuscript: "Perceptual Study of Semantically Synthesized 2D Chest CT Image Realism: Quantitative Metrics and Expert Opinion". 
Repository contains code for Computer Assisted Learning; data preprocessing, obtaining synthetic images and realism assessment.

## Data preprocessing
The publicly available dataset of [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) is used for this research and should be downloaded via their website (only select CT scans when downloading). 

`python preprocessing_CT.py`

Change input and output folder directories before running. Will convert the DICOM files into 2D original images, with corresponding semantic label map and visual label map. They will be stored in the folders 'image', 'label', and 'put' respectively. Examples can be seen below.

ADD IMAGE EXAMPLES

The pre-processed data can be divided into a combined training and validation set, and a test set. In the manuscript, subjects 1 through 900 are used for training validation and 901 through 1010 for testing. Additionally, for the experiments with a smaller training set, the first 2% of the full training set are used. In case of 0.3%, only subject 1 is used (without 20% sampling).

## Obtaining synthetic images
As semantic image synthesis network, the model developed by Park et al. is used, referred to as [SPADE](https://github.com/NVlabs/SPADE) or sometimes as GauGAN. 

ADD LINE HOW TO DOWNLOAD THEIR CODE
ADD LINE HOW TO TRAIN THE NETWORK

Their code requires the semantic label maps from 'label' as input, as well as the images from 'image' as guiding image. This is similar in both the training and testing phase. During testing the created synthetic images will be saved in an output directory. ADD THAT WE TRAINED ON A40 GPU's

ADD LINE OF CODE THAT WILL CREATE SYNTHETIC IMAGES FROM THE TEST SET

## Realism assessment
To calculate scores of quantitative metrics:

`python quantitative__metrics.py`

Change paths to synthetic and original images and indicate the metric to be applied before running. Possible metrics are; psnr, ssim, ms-ssim and lpips. 

ADD LINE HOW TO OBTAIN FID USING CODE FROM PARK

To set up a similar perceptual study:

`python qualitative_perceptual_study.py`

will create 60 folders with quartets of images, and saves information about their origin as .json. Change paths to output directory, and original and synthetic images before running.
