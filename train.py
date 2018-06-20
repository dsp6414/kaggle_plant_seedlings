'''
Create datasets, train last fc layer
then unfreeze model, apply differential learning rate to model
train full stack
'''
# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import CSV_Dataset
from generate_submission import gen_submission

#Add a ref to directory where train/test data is
data_dir = "/home/keith/data/plant_seedlings"

#where to save model checkpoint
cp_file = 'bestweights.ckpt'

#load the data
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load a model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, use_saved_weights = True):
    '''
    train model and save parameters
    :param model:
    :param criterion: loss function
    :param optimizer: how to optimize the loss function
    :param scheduler: lr scheduler
    :param num_epochs:
    :param use_saved_weights: if True will load the last best saved weights
    :return: the model with the best set of weights that have been found
    '''
    since = time.time()

    if (use_saved_weights == True):
        # load best model and optimizer
        checkpoint = torch.load(cp_file)
        model.load_state_dict(checkpoint['state_dict'])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the model checkpoint
                mod_opt_state = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(mod_opt_state, cp_file)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def freeze_first_n_layers(model, numb_layers):
    '''
    Freezing the first numb_layers of a model
    :param model:
    :param numb_layers:
    :return:
    '''
    for name, child in model.named_children():
        numb_layers -= 1
        for name2, params in child.named_parameters():
            params.requires_grad = (numb_layers <= 0)

        return model

def get_param_list(model, lr):
    '''
    apply learning rate list evenly to model layers
    return a parametized list of learning rates

    :param model:
    :param lr: list[the last is the fc layer learning rate, the rest are divided amongst the layers
    :return:
    '''
    numb_layers = 0
    for name, module in model_conv.named_children():
        if ('layer') in name:
            numb_layers += 1

    # get num layers per learning rate
    nlplr = numb_layers // (len(lr) - 1)

    params_dict = dict(model_conv.named_parameters())
    params = []
    for key, value in params_dict.items():
        # print(key)
        if key[:len("layer")] == "layer":
            # probably looks like layer1.0.xxxxx
            # get the layer1 bit

            layer_number = int(key[len("layer"):].split('.')[0])
            params += [{'params': [value], 'lr': lr[layer_number // nlplr]}]
        elif key[:len("fc")] == "fc":
            params += [{'params': [value], 'lr': lr[len(lr) - 1]}]
    return params

#######################################
#load pretrained model
model_conv = torchvision.models.resnet50(pretrained=True)

#freeze all but last layer
for param in model_conv.parameters():
    param.requires_grad = False

#configure fc layer
num_linear_inputs = model_conv.fc.in_features
num_outputs = 12 # number of weedlings
model_conv.fc = nn.Linear(num_linear_inputs, num_outputs)
model_conv = model_conv.to(device)

#model parameters
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model_conv.fc.parameters(), lr=0.001, weight_decay=1e-5)
exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs

# start with original model weights, train just fc layer
model_ft = train_model(model_conv, criterion, opt, exp_lr_scheduler, num_epochs=4, use_saved_weights = False)
gen_submission("preds_1.csv")

#######################################
#now train entire model (including fc layer) using differential learning rates
for param in model_conv.parameters():
    param.requires_grad = True

model_conv = model_conv.to(device)

lrs = [.0001, .00033, .001]     #for images similar to imagenet the learning rates can vary by *10, if not by about .33
                                #this dataset is masked, so different, so use .33
params = get_param_list(model_conv, lrs)    #apply lr evenly
opt = optim.Adam(params, lr=.0001, eps=1e-8, weight_decay=1e-5 )

# fine tune entire model
model_ft = train_model(model_conv, criterion, opt, exp_lr_scheduler,num_epochs=5)
gen_submission("preds_2.csv")

#######################################
#combine both datasets and train for 2 more epochs
print ("size training set before combo is" + str(len(image_datasets['train'])))
image_datasets['train'] =  image_datasets['train'] + image_datasets['val']  #the key bit, although val is useless here
print ("size training set after combo is" + str(len(image_datasets['train'])))

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# fine tune entire model with complete dataset
model_ft = train_model(model_conv, criterion, opt, exp_lr_scheduler,num_epochs=3)
gen_submission("preds_3.csv")

#######################################
#train with pseudolabeled data
train_dir = "/home/keith/data/plant_seedlings/train"
labels = sorted(next(os.walk(train_dir))[1])
csv_ds = CSV_Dataset(csv_file="preds.csv", root_dir="/home/keith/data/plant_seedlings/test/tst/",
                     possible_labels=labels, transform=data_transforms['train'])

print ("size training set before adding knowledge distillation test set is" + str(len(image_datasets['train'])))
image_datasets['train'] =  image_datasets['train'] + csv_ds  #the key bit
print ("size training set after combo is" + str(len(image_datasets['train'])))

model_ft = train_model(model_conv, criterion, opt, exp_lr_scheduler,num_epochs=2)
gen_submission("preds_4.csv")





