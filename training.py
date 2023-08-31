#importing required libraries
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

#to check if the system uses CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#Controling size of tensor so it is adjusted to control memory usage
torch.backends.cuda.max_split_size_mb = 1000

#Function to create a directory if it doesnt exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Dataset class
class PolypDataset(Dataset):
    def __init__(self, images_path, masks_path, image_transform=None, mask_transform=None, normalize=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.normalize = normalize
        self.n_samples = len(images_path)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #obtaining image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #obtaining mask
        
        #resizing image and mask
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # Convert numpy arrays to PIL Images
        image_pil = transforms.functional.to_pil_image(image)
        mask_pil = transforms.functional.to_pil_image(mask)
        
        #application of transforms if applicable
        if self.image_transform is not None:
            image_pil = self.image_transform(image_pil)
            image = np.array(image_pil)

        if self.mask_transform is not None:
            mask_pil = self.mask_transform(mask_pil)
            mask = np.array(mask_pil)
        
        #scaling    
        image= image/255.0
        mask=mask/255.0

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        if self.normalize is not None:
            image_tensor = self.normalize(image_tensor)

        return image_tensor, mask_tensor

# Define image and mask transforms
image_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1))

])

mask_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1))

])

#Initializing paths of image and mask 
image_path = sorted(glob('/kaggle/input/polypdatac1toc5/polypdata/train/image/*'))
mask_path = sorted(glob('/kaggle/input/polypdatac1toc5/polypdata/train/mask/*'))

#Finding mean and standard deviation of training images to use to normalize the images.
train_mean_sum = [0.0, 0.0, 0.0]
train_std_sum = [0.0, 0.0, 0.0]
num_train_samples = len(image_path)

for i in tqdm(range(num_train_samples), desc="Mean and std for training images"):
    image = cv2.imread(image_path[i], cv2.IMREAD_COLOR)
    image = image / 255.0  # Normalize image to [0, 1]

    for channel in range(3):  # Iterate over R, G, B channels
        train_mean_sum[channel] += image[:, :, channel].mean()
        train_std_sum[channel] += image[:, :, channel].std()

# Calculate mean and std for each channel for training images
train_mean = [train_mean_sum[channel] / num_train_samples for channel in range(3)]
train_std = [train_std_sum[channel] / num_train_samples for channel in range(3)]

print("Train Mean:", train_mean)
print("Train Std:", train_std)

normalize = transforms.Normalize(mean=train_mean, std=train_std)

#creating dataset 
Polyp_data = PolypDataset(image_path, mask_path, image_transform=image_transform_aug, mask_transform=mask_transform_aug, normalize=normalize)
#Polyp_data_noaug = PolypDataset(image_path, mask_path, normalize=normalize)

dataset_size = len(Polyp_data)
train_size = int(0.7 * dataset_size) #70% training data
val_size = int(0.2 * dataset_size) #20% validation data
test_size = dataset_size - (train_size + val_size) #10% test data

# Split the dataset
train_data, val_data, test_data = data.random_split(Polyp_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

#Dataloaders
train_loader = DataLoader(
        dataset=train_data,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

val_loader = DataLoader(
        dataset=val_data,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )

# Define an index for the sample you want to visualize
sample_index = 13

# Get the original image and mask
original_image = cv2.imread(image_path[sample_index], cv2.IMREAD_COLOR)
original_mask = cv2.imread(mask_path[sample_index], cv2.IMREAD_GRAYSCALE)

# Get the augmented image and mask
sample_image, sample_mask = Polyp_data.__getitem__(sample_index)

# Convert augmented image and mask to numpy arrays for plotting
augmented_image = np.transpose(sample_image.numpy(), (1, 2, 0))
augmented_mask = sample_mask.numpy().squeeze()

# Plot the images and masks
plt.figure(figsize=(10, 5))

# Original Image and Mask
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(original_mask, cmap="gray")
plt.title("Original Mask")
plt.axis("off")

# Augmented Image and Mask
plt.subplot(1, 4, 3)
plt.imshow(augmented_image)
plt.title("Augmented Image")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(augmented_mask, cmap="gray")
plt.title("Augmented Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

#Convolution class 
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

#UNet class inclusing encoder, decoder, bottleneck and classifier
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        #Contracting path   
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        #Bottleneck
        self.b = conv_block(512, 1024)

        #Expansive path
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        #classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        #Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        #Bottleneck
        b = self.b(p4)

        #Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

#creating a function to set a seed value for reproducability purposes
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#Function to compute the elapsed time per epoch during training
def epoch_time(start_time, end_time):
    time = end_time - start_time
    mins = int(time / 60)
    secs = int(time - (mins * 60))
    return mins, secs

#Class to compute the loss function for training
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

#Function to calculate the accuracy during training and validation
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
    accuracy = (num_correct/num_pixels)*100    
    return accuracy

#Function for training the model
def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    accuracy = []

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        #cache emptied for better memory utilization
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
                
    acc_score = check_accuracy(loader,model,device)        
    epoch_loss = epoch_loss/len(loader)

    return epoch_loss,acc_score

#Function for validating the model
def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    accuracy = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
             
            del x, y
            gc.collect()
            torch.cuda.empty_cache()

        acc_score = check_accuracy(loader,model,device)
        epoch_loss = epoch_loss/len(loader)
    return epoch_loss,acc_score

#Initilazing hyperparameters before training
seeding(42)
H = 512
W = 512
size = (H, W)
batch_size = 8
num_epochs = 25
lr = 1e-5
patience = 3
counter = 0

#creating path to store model checkpoints
checkpoint_path = '/kaggle/working/checkpoint.pth'
create_dir(checkpoint_path)

#iniilaizing arrays to store the accuarcy and loss values during each epoch
train_loss_graph=[]
train_acc_graph=[]
val_loss_graph=[]
val_acc_graph=[]

best_valid_loss = float("inf") #setting the value of validation loss to infinite initially
model = build_unet().to(device) #build a UNet model
#if model needs to be trained from the saved checkpoints uncomment the next two lines of code
#checkpoint = torch.load('/kaggle/input/model1e4/model_1e4.pth') #load the path with model checkpoints
#model.load_state_dict(checkpoint['model_state_dict']) #load the model state to the created UNet model for continuing training 

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
loss_fn = DiceBCELoss()


#TRAINING AND VALIDATION LOOP
for epoch in range(num_epochs):

    start_time = time.time()

    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
    valid_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
    
    train_loss_graph.append(train_loss)
    train_acc_graph.append(train_accuracy)
    val_loss_graph.append(valid_loss)
    val_acc_graph.append(val_accuracy)
    

    """ Saving the model """
    if valid_loss < best_valid_loss:
        data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
        print(data_str)

        best_valid_loss = valid_loss
        torch.save(model.state_dict(), checkpoint_path)

    else:
        counter += 1
        if counter >= patience:
            print("Early stopping at epoch", epoch+1)
            break


    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
    data_str += f'\tTrain Loss: {train_loss:.3f}\n'
    data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
    data_str += f'\t Train Accuracy: {train_accuracy:.3f}\n'
    data_str += f'\t Val. Accuracy: {val_accuracy:.3f}\n'
    print(data_str)

#save the trained model
torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
            }, '/kaggle/working/model.pth')

#function to plot the accuarcy and loss graphs
def plot_graph(train_loss_graph,train_acc_graph,val_loss_graph,val_acc_graph):
    
    np_val_acc_graph = np.array([x.cpu().numpy() for x in val_acc_graph])
    np_train_acc_graph = np.array([x.cpu().numpy() for x in train_acc_graph])
    # plot training and validation loss over epochs
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_graph, label='Training loss')
    plt.plot(val_loss_graph, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.show()

    # plot training and validation accuracy over epochs
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,2)
    plt.plot(np_train_acc_graph, label='Training accuracy')
    plt.plot(np_val_acc_graph, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy over Epochs')
    plt.show()

plot_graph(train_loss_graph,train_acc_graph,val_loss_graph,val_acc_graph)




