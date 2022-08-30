import os
import shutil
import cv2
import random
import json

# Define directories for the 3 trials, ground truth and output
syn1_dir = '...'    # synthetic images after training with 0.3% of full train set
syn2_dir = '...'    # synthetic images after training with 2% of full train set
syn3_dir = '...'    # synthetic images after training with full train set
gt_dir = '...'      # original images
out_dir = '...'     # output directory

# use fixed seed for reproducibility
random.seed(37)
# Get number=60 random cases or questions
number = 60
cases = os.listdir(syn1_dir)
choices = []
while len(choices) < number:
    choice = random.choice(cases)
    if choice in choices:
        continue
    else:
        choices.append(choice)

# For every choice in choices get image from the syn directories and gt. Place in outdir with question number in random
# order and save this order in a file. For instance: question 1[img1=syn3, img2=gt, img3=syn2, img4=syn1].

names_list = ['a', 'b', 'c', 'd']
questionnaire = {}
for i in range(0, len(choices)):
    # load images
    img1 = cv2.imread((os.path.join(syn1_dir, choices[i])))
    img2 = cv2.imread((os.path.join(syn2_dir, choices[i])))
    img3 = cv2.imread((os.path.join(syn3_dir, choices[i])))
    img4 = cv2.imread((os.path.join(gt_dir, choices[i])))
    # create folder for current image number
    if not os.path.exists(os.path.join(out_dir, 'question' + str(i + 1))):
        os.makedirs(os.path.join(out_dir, 'question' + str(i + 1)))
    # create dict for question
    # copy images to question folder with random index and create tuple for question
    random.seed()
    random.shuffle(names_list)
    shutil.copy(syn1_dir + choices[i], out_dir + 'question' + str(i + 1) + '/' + names_list[0] + '.png')
    first = {'id': names_list[0], 'name': choices[i], 'class': 'trial1'}
    shutil.copy(syn2_dir + choices[i], out_dir + 'question' + str(i + 1) + '/' + names_list[1] + '.png')
    second = {'id': names_list[1], 'name': choices[i], 'class': 'trial2'}
    shutil.copy(syn3_dir + choices[i], out_dir + 'question' + str(i + 1) + '/' + names_list[2] + '.png')
    third = {'id': names_list[2], 'name': choices[i], 'class': 'trial3'}
    shutil.copy(gt_dir + choices[i][:-3] + 'jpg', out_dir + 'question' + str(i + 1) + '/' + names_list[3] + '.jpg')
    fourth = {'id': names_list[3], 'name': choices[i], 'class': 'GT'}
    question = first, second, third, fourth
    # Add question to dict
    questionnaire['question' + str(i + 1)] = question

# save dictionary with info on questions
with open("questionnaire.json", "w") as f:
    json.dump(questionnaire, f)
