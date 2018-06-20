
# imports
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import glob
import copy

#Pseudolabeled DataLoader
#takes datasets consisting of:
#   all training and validation data (trainvaldata)
#   pseudolabled prediction data (pseudolableddata)
#   provides batches consisting of 2/3 trainvaldata and 1/3 pseudolabeleddata
#pseudolabeleddata must be generated before can be loaded into a dataset, use generate_submission to get csv labling
#this type of thing is also called knowledge distillation
class CSV_Dataset(Dataset):
    """Generate a dataset from a csv file of the form
    file,species
    25cf6eb73.png,Maize
    953496deb.png,Fat Hen

    where the first param is the name of the file, thesecond is the prediction
    """

    def __init__(self, csv_file, root_dir, possible_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            possible_labels (list): a list of all the possible labels in the csv file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.label_lookup = {}
        for i,lab in enumerate(sorted(possible_labels)):
            self.label_lookup[lab]=i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #get the image
        filename = os.path.join(self.root_dir,self.data.iloc[idx, 0])
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)

        #translate string label to number
        lbl = self.label_lookup[self.data.iloc[idx, 1]]
        #return image and its label
        return img, lbl

class Unlabeled_Dataset(Dataset):
    '''
    provides access to unlabeled files for testing
    '''
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        images = []
        # for filename in sorted(glob.glob(root + "*.png")):  #gets entire path+filename
        for filename in os.listdir(root_dir):                 #just filename
            images.append('{}'.format(filename))

        self.root_dir = root_dir
        self.imgs = images
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        img = Image.open(os.path.join(self.root_dir, filename))
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ul_ds = Unlabeled_Dataset(root_dir ="/home/keith/data/plant_seedlings/test/tst/", transform=data_transforms)

    #how many images
    for i in range(len(ul_ds)):
        print(ul_ds[i][1])
    for i, (img, fle) in enumerate(ul_ds):
        print( fle )

    #test the loader
    dataloader = DataLoader(ul_ds, batch_size=4, shuffle=False, num_workers=4)

    for i, (imgs, fles) in enumerate(dataloader):
        print(imgs, fles )
        print("------------------")

    train_dir = "/home/keith/data/plant_seedlings/train"
    labels = sorted(next(os.walk(train_dir))[1])
    csv_ds = CSV_Dataset(csv_file = "test_preds.csv", root_dir="/home/keith/data/plant_seedlings/test/tst/",possible_labels = labels, transform=data_transforms)
    print ("length of dataset is " + str(len(csv_ds)))
    for i, (img, lbl) in enumerate(csv_ds):
        print( "label is "+ str(lbl))



