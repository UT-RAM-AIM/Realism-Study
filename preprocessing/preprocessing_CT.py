import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import pydicom
import os
import pylidc as pl
from pylidc.utils import consensus
from PIL import Image as im

from segmentations import semantic_nonROI, depict_semantic_lung


def init_(input_folder_, patient_id_, slices_):
    k_ = 'LIDC-IDRI-%s' % patient_id_
    patients_ = input_folder_ + '/' + k_
    date = os.listdir(patients_)
    date.sort()

    # To pick a specific scan (date)
    for k in range(0, len(date)):
        # Skip if encounter non-folder objects
        if date[k].startswith('.'):
            continue
        else:
            date_ = patients_ + '/' + date[k]
            # List all categories of that specific date scan
            branch = os.listdir(date_)
            branch.sort()
            # only use first patient scan
            # Skip if encounter non-folder objects
            if branch[0].startswith('.'):
                continue
            else:
                branch_ = date_ + '/' + branch[0]
                # read files
                slices_ = [pydicom.read_file(branch_ + '/' + s) for s in os.listdir(branch_) if s.endswith('.dcm')]
                slices_.sort(key=lambda x: int(x.InstanceNumber))

    return slices_, k_


def nodule_list(ann_, num_, all_slices, patient_id_):
    slist_, cmlist_, cblist_, inlist_, mlist_ = [], [], [], [], []
    lengthy_ = len(all_slices)
    direction = 'TBD'
    # Some cases go from top to bottom, others the other way. Should account for this with nodule annotations
    start_end_zval = [all_slices[0].ImagePositionPatient[-1], all_slices[-1].ImagePositionPatient[-1]]
    if start_end_zval[0] > start_end_zval[1]:
        direction = 'TOP_TO_BOTTOM'
    elif start_end_zval[0] < start_end_zval[1]:
        direction = 'BOTTOM_TO_TOP'
        print('The following case is upside down: ', patient_id_)
    else:
        print('Please check scan direction of patient ', patient_id_)

    for j_ in range(0, len(ann_)):
        if len(ann_[j_]) >= num_:  # check if number of annotations is >= number of consensus radiologists defined
            nods_ = ann_[j_]

            # Averaged malignancy score
            malig_ = []
            for i_ in range(0, len(ann_[j_])):
                malig_.append(ann_[j_][i_].malignancy)
            malig_ = np.median(malig_)
            print("Malignancy: %s" % malig_)

            # 50% consensus for nodule mask
            cmask, cbbox, masks = consensus(nods_, clevel=0.5, pad=[(512, 512), (512, 512), (0, 0)])
            for slice_ in range(cbbox[2].start, cbbox[2].stop):
                index_ = slice_ - cbbox[2].start
                contour = cmask[:, :, index_]
                x_ = False
                for j in contour:
                    if j.any():
                        x_ = True
                if x_:
                    if direction == 'BOTTOM_TO_TOP':
                        slist_.append(slice_)  # Some scans are saved upside down
                    elif direction == 'TOP_TO_BOTTOM':
                        slist_.append(lengthy_ - slice_ - 1)
                    else:
                        print('Please check scan direction of patient ', patient_id_)
                    cmlist_.append(cmask)
                    cblist_.append(cbbox)
                    inlist_.append(index_)
                    mlist_.append(malig_)

    list_ = list(dict.fromkeys(slist_))

    return list_, slist_, cmlist_, cblist_, inlist_, mlist_


def rescale_intercept(image_, slope_, intercept_):
    # Rescale
    image_ = image_.astype(np.int16)
    # remove pixel values outside of scan
    image_[image_ <= -2000] = 0

    # slope
    if slope_ != 1:
        image_ = slope_ * image_.astype(np.float64)
        image_ = image_.astype(np.int16)

    # Intercept
    image_ += np.int16(intercept_)

    # HU Clip
    image_[image_ < -1350] = -1350
    image_[image_ > 150] = 150

    return image_


def save_figure(img, label_type, slice_num, patient_id_, output_folder_, my_dpi=600):
    # check if three folders exist and save image, label, put to that folder
    if label_type == 'gt':
        output_folder = output_folder_ + '/image/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        file = patient_id_ + '-%s.png' % slice_num
        # 665 or my_dpi might need to be adjusted depending on your screen resolution
        plt.figure(figsize=(665 / my_dpi, 665 / my_dpi), dpi=my_dpi)
        plt.imshow(img, cmap=plt.cm.gray, vmin=img.min(), vmax=img.max())
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
        plt.close()
    elif label_type == 'semantic':
        output_folder = output_folder_ + '/label/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        file = patient_id_ + '-%s.png' % slice_num
        label = im.fromarray(img)
        label = label.convert("L")
        label.save(output_folder + file)
    elif label_type == 'semantic_color':
        # cmap in manuscript review:
        # cmap = matplotlib.colors.ListedColormap(['#000000', '#08094B', '#52648e', '#7ba4e9', '#86ac41', '#fbb41a'])
        # cmap in manuscript ps:
        # cmap = matplotlib.colors.ListedColormap(['black', 'darkblue', 'slateblue', 'darkcyan',
        # 'mediumaquamarine', 'yellow'])
        output_folder = output_folder_ + '/put/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        file = patient_id_ + '-%s.png' % slice_num
        # 665 or my_dpi might need to be adjusted depending on your screen resolution
        plt.figure(figsize=(665 / my_dpi, 665 / my_dpi), dpi=my_dpi)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
        plt.close()


def depict_semantic_nodule(image_, patient_id_, temp_, semantic_map_, output_folder_, list_, cmlist_, inlist_, mlist_):
    # patient_id_: patient's id number, 0001->1
    # slices_: numbers of current slices
    # num_: numbers of annotations that radiologists agreed with
    # semantic_map_: semantic label map include lung and other parts

    for w in range(0, len(list_)):
        if list_[w] == temp_:
            contour = cmlist_[w][:, :, inlist_[w]]
            # Benign
            if mlist_[w] < 2.5:
                semantic_map_[contour == True] = 5
            # Indeterminate
            elif 2.5 <= mlist_[w] < 3.5:
                semantic_map_[contour == True] = 6
            # Malignant
            elif mlist_[w] >= 3.5:
                semantic_map_[contour == True] = 7
            else:
                print("Unknown malignancy: %s" % mlist_[w])

    semantic_map_ = semantic_map_.astype(np.int32)
    # Save Ground Truth ###
    save_figure(image_, 'gt', temp_, patient_id_, output_folder_, my_dpi=600)

    # Semantic Processing ###
    segment_map_ = np.zeros((semantic_map_.shape[0], semantic_map_.shape[0]), np.uint8)
    segment_map_[semantic_map_ == 0] = 0
    segment_map_[semantic_map_ == 1] = 1
    segment_map_[semantic_map_ == 2] = 2
    segment_map_[semantic_map_ == 3] = 3
    segment_map_[semantic_map_ == 4] = 4
    segment_map_[semantic_map_ == 5] = 5
    segment_map_[semantic_map_ == 6] = 6    # change to 5 for no differentiation in nodule type
    segment_map_[semantic_map_ == 7] = 7    # change to 5 for no differentiation in nodule type

    # Save Semantic Label ###
    save_figure(segment_map_, 'semantic', temp_, patient_id_, output_folder_, my_dpi=600)

    # Save Colorized Semantic Label ##
    save_figure(segment_map_, 'semantic_color', temp_, patient_id_, output_folder_, my_dpi=600)


def selection(nodule_, len_slices_):
    # 20% of current patients
    lengthy_ = len_slices_ // 5
    if len(nodule_) >= lengthy_:
        print("Nodule list is greater than the amount of selected slices")
    # declare the returned list: select_
    select_ = copy.deepcopy(nodule_)
    # Start sampling between the first 10 slices and move up 10 slices every iter
    low_ = 0
    high_ = 9
    iter_ = 0

    while len(select_) < lengthy_:
        temp1_ = random.randint(low_, high_)
        if temp1_ < len_slices_:
            if not (temp1_ in select_):
                select_.append(temp1_)
                iter_ = 0
                if (len_slices_ - temp1_) > 10:
                    low_ += 10
                    high_ += 10
                else:
                    low_ = 0
                    high_ = 9
            # If the current interval are all selected
            elif iter_ > 10:
                if (len_slices_ - temp1_) > 10:
                    low_ += 10
                    high_ += 10
                else:
                    low_ = 0
                    high_ = 9
            else:
                iter_ += 1
    select_.sort()
    print("Finish selection, target slices %d, and we got %d" % (lengthy_, len(select_)))

    return select_


def depiction_(temp_, slices_, class1_, class2_, class3_, region_threshold_, lower_, region_threshold2_, id_,
               output_folder_, nodule_, cmlist_, inlist_, mlist_):
    # d the variable to store slice information, temp_ current slice number
    sslice = slices_[temp_]

    # Rescale, Intercept and HU Clip
    image_ = rescale_intercept(sslice.pixel_array, sslice.RescaleSlope, sslice.RescaleIntercept)

    # Generation of semantic labels for Class1, 2 and 3; body, low dense, high dense
    semantic_map_, image_ = semantic_nonROI(image_, class1_[0], class2_[0], class3_[0], class3_[1])

    # Semantic label for Class 4, the lung region
    semantic_map_1_ = depict_semantic_lung(image_, semantic_map_, region_threshold_, lower_, region_threshold2_)
    # If there is no lung label, continue; we are only interested in slices with lung
    if 4 in semantic_map_1_:
        # Semantic label for Class 5, the nodule
        depict_semantic_nodule(image_, id_, temp_, semantic_map_1_, output_folder_, nodule_, cmlist_, inlist_, mlist_)
    else:
        print('Slice %s from LIDC-IDRI-%s does not contain any lung area; it is skipped' % (temp_, id_))


def create_labels(patient_id_, input_folder_, radiologists_, class1_, class2_, class3_, region_threshold_,
                  region_threshold2_, lower_, output_folder_, use_all_data=True):
    # Create container for all slices and nodule slices
    slices_ = []

    # quickly analyze scan/annotations and continue if slice thickness > 2.5 mm
    scan_ = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-%s' % patient_id_).first()
    ann_ = scan_.cluster_annotations()

    if scan_.slice_thickness <= 2.5:
        # Initialize DICOM data
        slices_, id_ = init_(input_folder_, patient_id_, slices_)

        # Initialize nodule slices and consensus mask
        list_, nodule_, cmlist_, cblist_, inlist_, mlist_ = nodule_list(ann_, radiologists_, slices_, patient_id_)

        if not use_all_data:
            slice_nums_ = selection(nodule_, len(slices_))  # selects 20% of patient scan including all nodule slices
            for temp_ in slice_nums_:
                depiction_(temp_, slices_, class1_, class2_, class3_, region_threshold_, lower_, region_threshold2_,
                           id_, output_folder_, nodule_, cmlist_, inlist_, mlist_)
        else:
            # Generation of semantic labels
            for temp_ in range(0, len(slices_)):
                depiction_(temp_, slices_, class1_, class2_, class3_, region_threshold_, lower_, region_threshold2_,
                           id_, output_folder_, nodule_, cmlist_, inlist_, mlist_)
    else:
        print("LIDC-IDRI-%s is skipped due to slice thickness greater than 2.5mm" % patient_id_)


##
#################################
# Parameters and initialization #
#################################
INPUT_FOLDER = '...'  # LIDC-IDRI folder
OUTPUT_FOLDER = '...'  # Output folder.
class1 = [-400, 150]  # HU clip for the background
class2 = [5, 145]  # HU clip for soft tissues
class3 = [145, 150]  # HU clip for high dense tissues
# Lower bound of lung area. If setting it to 0, some noises will be included and mess up the semantic labels.
region_threshold = 1100
# Higher bound of lun area. Tuning this value would impact the output: decreasing the number would skip true lung,
# while increasing would include the background.
region_threshold2 = 230000
radiologists = 3  # 3 radiologists consensus is used
# Similarity threshold between Kmeans and FindContours. Normally to set 0.7, but if you want diagnostic mode set it to 0
lower = 0.7
all_data = False  # if False, 20% of all slices per scan (including nodule slices) will be used

# Iteratively append the patient numbers. i.e. patient ID: 20 -> 0020; patient ID: 1 -> 0001
# Determine range of patient IDs to run in one go
run_list = []
# change to start and stop of preference
start = 1
stop = 100
while start <= stop:
    if start < 10:
        num = '000%s' % start
    elif 10 <= start < 100:
        num = '00%s' % start
    elif 100 <= start < 1000:
        num = '0%s' % start
    else:
        num = '%s' % start
    run_list.append(num)
    start += 1

# Preprocess and Generation of semantic labels and save lung nodule annotations/locations
for patient_id in run_list:
    create_labels(patient_id, INPUT_FOLDER, radiologists, class1, class2, class3, region_threshold, region_threshold2,
                  lower, OUTPUT_FOLDER, all_data)
