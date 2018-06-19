
# imports
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import glob
import copy



class TestImageFolder(Dataset):
    '''
    provides access to unlabeled files for testing
    '''
    def __init__(self, root, transforms=None):
        images = []
        # for filename in sorted(glob.glob(root + "*.png")):  #gets entire path+filename
        for filename in os.listdir(root):                 #just filename
            images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transforms = transforms

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        img = Image.open(os.path.join(self.root, filename))
        if self.transforms:
            img = self.transforms(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_ds = TestImageFolder(root = "/home/keith/data/plant_seedlings/test/tst/", transforms=data_transforms)

    #how many images
    # for i in range(len(test_ds)):
    #     print(test_ds[i][1])

    # for i, (img, fle) in enumerate(test_ds):
    #     print( fle )

    dataloader = DataLoader(test_ds, batch_size=4,shuffle=False, num_workers=4)

    for i, (imgs, fles) in enumerate(dataloader):
        print(imgs, fles )
        print("------------------")

