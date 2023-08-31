'''Assuming that the model, dataloader and dataset has been created or is available. This file is a continuation of the training.py and should be implemented after training.py
The files were created seperately for the easy readability for the user.'''

#Import required libraries
import os
import numpy as np
import cv2
import random
import gc
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import os, time
from operator import add
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from scipy.ndimage import distance_transform_edt

#Function that calculates the metrices jaccard score, f1 score, recall, precision and accuracy  
#from the ground truth and predicitons
def calculate_metrics(y_true, y_pred):
    #Ground truth 
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    #Prediction 
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    jaccard = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return [jaccard, f1, recall, precision, accuracy]

#function to calculate the surface distances
def surface_distance(mask1, mask2, spacing):

    # Calculate distance transform for each mask
    dt1 = distance_transform_edt(mask1, spacing)
    dt2 = distance_transform_edt(mask2, spacing)
    
    # Convert boolean masks to integer masks (0 or 1)
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    
    # Calculate the surface distances
    surface_dist = np.abs(dt1 - dt2) * (mask1 | mask2)

    return surface_dist

def calculate_surface_distances(pred_mask, gt_mask):
    spacing = 1.0  # Spacing between pixels in millimeters

    surface_dist = surface_distance(pred_mask, gt_mask, spacing)
    return surface_dist
  
#function to handle mask
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

''' TESTING FOR TEST SET THAT WAS CREATED DURING THE SPLIT OF DATASET INITIALLY '''

#test dataloader
test_loader = DataLoader(
    dataset=test_data,
    batch_size=8,
    shuffle=False,  # No need to shuffle for evaluation
    num_workers=2
)
#load the model to be tested
model = build_unet()
model = model.to(device)
checkpoint = torch.load('/kaggle/input/model-final/model.pth') #checkpoint path 
model.load_state_dict(checkpoint['model_state_dict'])

# Calculating Jaccard, F1, Recall, Precision, Accuracy
metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]  
time_taken = []

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for x, y in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device)
        y = y.to(device)

        start_time = time.time()
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        score = calculate_metrics(y, pred_y)
        metrics_score = list(map(add, metrics_score, score))
        

# Calculate and print metrics
jaccard = metrics_score[0] / len(test_loader)
f1 = metrics_score[1] / len(test_loader)
recall = metrics_score[2] / len(test_loader)
precision = metrics_score[3] / len(test_loader)
acc = metrics_score[4] / len(test_loader)
print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

#calculate surface distances 
surface_distances = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred_y = model(x)
        pred_masks = pred_y.cpu().numpy() > 0.5
        y_masks = y.cpu().numpy()

        for pred_mask, gt_mask in zip(pred_masks, y_masks):
            surface_dist = surface_distance(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0))
            surface_distances.append(surface_dist)

# Convert the list of arrays into a single numpy array
surface_distances = np.concatenate(surface_distances)

# Calculate mean, median, and standard deviation of surface distances
mean_surface_distance = np.mean(surface_distances)
median_surface_distance = np.median(surface_distances)
std_surface_distance = np.std(surface_distances)

print("Mean Surface Distance:", mean_surface_distance)
print("Median Surface Distance:", median_surface_distance)
print("Standard Deviation of Surface Distance:", std_surface_distance)


'''GENERALISABILITY TEST ON UNSEEN DATA'''

#Seeding
seeding(42)

#Folders created to store the sample tested image, ground truth and predictions
create_dir("/kaggle/working/results1/")

#Load dataset
test_x = sorted(glob("/kaggle/input/polypdatac1toc5/polypdata/test/image_test/*"))
test_y = sorted(glob("/kaggle/input/polypdatac1toc5/polypdata/test/mask_test/*"))

H = 512
W = 512
size = (W, H)

metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
time_taken = []

for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):

    
    name = x.split("/")[-1].split(".")[0]

    #Reading image
    image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    #Reading mask
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    mask = cv2.resize(mask, size)
    y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
    y = y/255.0
    y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)
    y = y.to(device)

    with torch.no_grad():
        
        start_time = time.time()
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        total_time = time.time() - start_time
        time_taken.append(total_time)


        score = calculate_metrics(y, pred_y)
        metrics_score = list(map(add, metrics_score, score))
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    
    ori_mask = mask_parse(mask)
    pred_y = mask_parse(pred_y)
    line = np.ones((size[1], 10, 3)) * 128

    cat_images = np.concatenate(
        [image, line, ori_mask, line, pred_y * 255], axis=1
    )
    cv2.imwrite(f"results1/{name}.jpg", cat_images)

jaccard = metrics_score[0]/len(test_x)
f1 = metrics_score[1]/len(test_x)
recall = metrics_score[2]/len(test_x)
precision = metrics_score[3]/len(test_x)
acc = metrics_score[4]/len(test_x)
print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

fps = 1/np.mean(time_taken)
print("FPS: ", fps)

surface_distances = []

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # Reading mask
        gt_mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
        gt_mask = cv2.resize(gt_mask, size)
        gt_mask = gt_mask > 0.5
        
        # Reading image
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)
        image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        # Forward pass
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_mask = pred_y[0][0].cpu().numpy() > 0.5  # Assuming batch size is 1

        surface_dist = calculate_surface_distances(pred_mask, gt_mask)
        surface_distances.append(surface_dist)

# Convert the list of arrays into a single numpy array
surface_distances = np.concatenate(surface_distances)

# Calculate mean, median, and standard deviation of surface distances
mean_surface_distance = np.mean(surface_distances)
median_surface_distance = np.median(surface_distances)
std_surface_distance = np.std(surface_distances)

print("Mean Surface Distance:", mean_surface_distance)
print("Median Surface Distance:", median_surface_distance)
print("Standard Deviation of Surface Distance:", std_surface_distance)










