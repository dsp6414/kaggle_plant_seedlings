
# imports
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
from utils import Unlabeled_Dataset
import time

#Add a ref to directory where train/test data is
data_dir = "/home/keith/data/plant_seedlings"
train_dir = "/home/keith/data/plant_seedlings/train"        #directories of particular images
test_dir = "/home/keith/data/plant_seedlings/test/tst/"     #a mix of images
# test_dir = "/home/keith/data/plant_seedlings/train/Shepherds Purse/"  #used for verification

#where to get model checkpoint
cp_file = 'bestweights.ckpt'


def gen_submission(pred_file = "preds.csv"):
    '''

    :param pred_file: where to write the predictions
    :return:
    '''

    # get a sorted list of directories
    # count how many files in each folder
    data_dir_train = os.path.join(data_dir, 'train')
    labels = sorted(next(os.walk(data_dir_train))[1])

    # only the transforms relevant to predictions
    test_data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_ds = Unlabeled_Dataset(root_dir=test_dir, transform=test_data_transform)
    dataset_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t1 = time.time()

    # load pretrained model and change fc layer to have 12 outputs
    model_pred = torchvision.models.resnet50(pretrained=True)
    num_linear_inputs = model_pred.fc.in_features
    num_outputs = 12  # number of weedlings
    model_pred.fc = nn.Linear(num_linear_inputs, num_outputs)
    model_conv = model_pred.to(device)
    criterion = nn.CrossEntropyLoss()
    # opt = optim.Adam(model_pred.fc.parameters(), lr=0.001, weight_decay=1e-5)
    # exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs
    # load best model
    checkpoint = torch.load(cp_file)
    model_pred.load_state_dict(checkpoint['state_dict'])
    # we are evaluating now
    model_pred.eval()
    # we are not doing any gradients on predictions
    preds_file = open(pred_file, "w")
    preds_file.write('file,species\n')
    with torch.no_grad():
        for images, filenames in dataset_loader:
            images = images.to(device)
            outputs = model_pred(images)
            _, predicted = torch.max(outputs.data, 1)
            for file, pred in zip(filenames, predicted):
                preds_file.write(file + "," + labels[pred] + '\n')

    t2 = time.time() - t1
    print("total time elapsed " + str(t2))

if __name__ == "__main__":
    gen_submission()