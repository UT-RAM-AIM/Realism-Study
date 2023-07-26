import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image as im
from skimage import measure


def manipulate_move(tuple_, map_, malig_, hor=-300, ver=100):
    map2_ = map_.copy()
    map3_ = map_.copy()
    nodule_ = np.zeros((map_.shape[0], map_.shape[0]), np.uint8)
    nodule_[tuple_] = malig_

    # adjust nodule map to put it at a different position
    if hor != 0:
        if hor < 0:  # means moving nodule to the right.
            nodule_ = np.c_[np.zeros((np.shape(map_)[0], np.abs(hor))), nodule_]
            nodule_ = im.fromarray(nodule_)
            nodule_ = nodule_.crop((0, 0, map_.shape[0], map_.shape[1]))
        else:
            nodule_ = np.c_[nodule_, np.zeros((np.shape(map_)[0], np.abs(hor)))]
            nodule_ = im.fromarray(nodule_)
            nodule_ = nodule_.crop((np.abs(hor), 0, map_.shape[0] + np.abs(hor), map_.shape[1]))
        nodule_ = np.array(nodule_)
    if ver != 0:
        if ver < 0:  # means moving the nodule down
            nodule_ = np.c_[np.zeros((np.shape(map_)[0], np.abs(ver))), nodule_]
            nodule_ = im.fromarray(nodule_)
            nodule_ = nodule_.crop((0, 0, map_.shape[0], map_.shape[1]))
        else:
            nodule_ = np.c_[nodule_, np.zeros((np.shape(map_)[0], np.abs(ver)))]
            nodule_ = im.fromarray(nodule_)
            nodule_ = nodule_.crop((0, np.abs(ver), map_.shape[0], map_.shape[1] + np.abs(ver)))
        nodule_ = np.array(nodule_)

    map2_[nodule_ != 0] = malig_
    map2_[map3_ == 0] = 0
    map2_[map3_ == 1] = 1
    map2_[map3_ == 2] = 2
    map2_[map3_ == 3] = 3
    plt.figure()
    plt.imshow(map2_)
    plt.show()

    return map2_


def manipulation_expand(tuple_, map_, size_, malig_):
    map2_ = map_.copy()
    map3_ = map_.copy()
    nodule_ = np.zeros((map_.shape[0], map_.shape[0]), np.uint8)
    nodule_[tuple_] = malig_

    img_ = im.fromarray(nodule_)
    img_ = img_.resize((size_, size_))
    temp_ = np.array(img_)
    x = np.where(temp_ == malig_)  # print(map_.max())
    # Sometimes it occurs that the image is reduced in size too much. So much, that the nodule disappears. In that case,
    # increase the size again until the nodule is visible again
    while np.size(x) == 0:
        size_ = size_ * 2
        img_ = im.fromarray(nodule_)
        img_ = img_.resize((size_, size_))
        temp_ = np.array(img_)
        x = np.where(temp_ == malig_)
    left_ = int(x[1].mean() - tuple_[1].mean())
    bottom_ = int(x[0].mean() - tuple_[0].mean())
    img_ = img_.crop((left_, bottom_, left_ + 512, bottom_ + 512))
    temp_ = np.array(img_)
    plt.figure()
    plt.imshow(temp_)
    plt.show()
    # when increasing the size can cause overlap with other structures
    map2_[temp_ != 0] = malig_
    map2_[map3_ == 0] = 0
    map2_[map3_ == 1] = 1
    map2_[map3_ == 2] = 2
    map2_[map3_ == 3] = 3

    plt.figure()
    plt.imshow(map2_)
    plt.show()

    return map2_


def calculate_expansion(tup_, seg_map_, malignancy_, dist_):
    map1_ = seg_map_.copy()
    nodule_ = np.zeros((map1_.shape[0], map1_.shape[0]), np.uint8)
    nodule_[tup_] = malignancy_
    labels_ = measure.label(nodule_)
    props_ = measure.regionprops(labels_)
    new_diam_ = dist_

    for k in range(0, len(props_)):
        bbox_start_ = props_[k]['bbox'][0]
        if bbox_start_ == tup_[0][0]:
            while not new_diam_:
                if malignancy_ == 5:
                    diam_ = np.random.normal(16.41, 10.02)  # np.random.normal(16.41, 10.02)
                    if diam_ > 0:
                        new_diam_ = diam_
                elif malignancy_ == 7:
                    diam_ = np.random.normal(29.39, 13.28)
                    if diam_ > 0:
                        new_diam_ = diam_
    if len(props_) > 1:
        old_diam_ = []
        for m in range(0, len(props_)):
            old_diam_.append(props_[m]['equivalent_diameter'])
        shape_change_ = new_diam_ / np.max(old_diam_)
        print('Multiple contours, used largest diameter of %s' % (np.max(old_diam_)))
    else:
        shape_change_ = new_diam_ / props_[0]['equivalent_diameter']

    new_size_ = int(np.round(shape_change_ * 512))

    return new_size_


def get_map_malignancy(input_dir, label_name, update_map=np.zeros((512, 512))):
    # load a segmentation map
    if update_map.any() != 0:
        sem_np = update_map
    else:
        sem_ = im.open(os.path.join(input_dir, label_name))
        sem_np = np.asarray(sem_)

    # Create new segmenation map without nodule
    segment_wonod_ = np.zeros((sem_np.shape[0], sem_np.shape[0]), np.uint8)
    segment_wonod_[sem_np == 0] = 0
    segment_wonod_[sem_np == 1] = 1
    segment_wonod_[sem_np == 2] = 2
    segment_wonod_[sem_np == 3] = 3
    segment_wonod_[sem_np == 4] = 4
    segment_wonod_[sem_np == 5] = 4
    segment_wonod_[sem_np == 6] = 4
    segment_wonod_[sem_np == 7] = 4

    # Create new segmentation map with nodule malignancy
    segment_map_ = np.zeros((sem_np.shape[0], sem_np.shape[0]), np.uint8)
    segment_map_[sem_np == 0] = 0
    segment_map_[sem_np == 1] = 1
    segment_map_[sem_np == 2] = 2
    segment_map_[sem_np == 3] = 3
    segment_map_[sem_np == 4] = 4
    segment_map_[sem_np == 5] = 5  # benign
    segment_map_[sem_np == 6] = 6  # indeterminate
    segment_map_[sem_np == 7] = 7  # malignant

    # search for instances of malignancy label
    malignant_nodule = np.where(segment_map_ == 7)
    benign_nodule = np.where(segment_map_ == 5)
    indeterminate_nodule = np.where(segment_map_ == 6)

    # check which malignancy labels occur in semantic map
    exist1 = 7 in segment_map_
    exist2 = 5 in segment_map_
    exist3 = 6 in segment_map_

    if exist1:  # malignant nodule
        print('Malignant nodule in label map.')
    elif exist2:    # benign nodule
        print('Benign nodule in label map.')
    elif exist3:    # indeterminate nodule
        print('Indeterminate nodule in map.')

    return segment_wonod_, malignant_nodule, benign_nodule, indeterminate_nodule


def save_updated_labels(map_, out_folder, seg_name, my_dpi=600):

    # Semantic Label
    output_folder2 = out_folder + 'label/'
    if not os.path.exists(output_folder2):
        os.mkdir(output_folder2)
    file2 = seg_name
    label = im.fromarray(map_)
    label = label.convert("L")
    label.save(output_folder2 + file2)

    # Colorized Semantic Label
    output_folder3 = out_folder + '/put/'
    if not os.path.exists(output_folder3):
        os.mkdir(output_folder3)
    plt.figure(figsize=(665 / my_dpi, 665 / my_dpi), dpi=my_dpi)
    plt.imshow(map_, vmin=0, vmax=7)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder3, file2), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()


label_ = 'D:/SIS/train_data/Data5/test/label/'
output_folder_ = 'D:/SIS/train_data/Data5/test/'

label_ls = os.listdir(label_)
label_ls.sort()

# Create distributions for malignant and benign resizing
benign_dist = np.clip(np.random.gamma(3.3489, 4.8997, 450), 2, 43)
malignant_dist = np.clip(np.random.normal(29.389, 13.285, 150), 2, 60)
# get random samples from the corresponding size distribution above
benign = np.random.choice(benign_dist)
malignant = np.random.choice(malignant_dist)

# Get label name of map you want to manipulate:
name_ = label_ls[60]
# Get semantic map and nodule information
seg_map, mal, ben, indet = get_map_malignancy(label_, name_)
# manipulate nodule labels
if mal[0].any() != 0:
    # resizing and moving malignant nodule
    print('Resizing or moving malignant nodule.')
    # change malignant to random number within distribution
    size_adjust_ = calculate_expansion(mal, seg_map, 7, malignant)
    seg_map = manipulation_expand(mal, seg_map, size_adjust_, 7)
    # or move the nodule (uncomment below). Need to update nodule label so re-run get_map_malignancy first
    # seg_map, mal, ben, indet = get_map_malignancy(label_, name_, update_map=seg_map)
    # seg_map = manipulate_move(mal, seg_map, 7)   # possible to indicate horizontal and vertical displacement
if ben[0].any() != 0:
    # resizing and moving benign nodule
    print('Resizing or moving benign nodule.')
    # change benign to random number within distribution
    size_adjust_ = calculate_expansion(ben, seg_map, 5, benign)
    seg_map = manipulation_expand(ben, seg_map, size_adjust_, 5)
    # if moving is also wanted, see above at malignancy
if indet[0].any() != 0:
    # For now we didn't do anything with indeterminate, except remove it from the map. We could of course do something
    # similar as we do with the nodules above.
    print('Removing indeterminate nodule...')

# save new semantic label map (annotation map)
save_updated_labels(seg_map, output_folder_, name_)
