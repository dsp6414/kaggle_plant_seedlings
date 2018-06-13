# imports

import os
import operator
from shutil import copyfile


#Add a ref to directory where train/test data is
data_dir = "/home/keith/data/plant_seedlings/"
data_dir_train = os.path.join(data_dir,'train')
data_dir_val = os.path.join(data_dir,'val')

val_percent  = .2

#count how many files in each folder
allfiles = {}
for root, dirs, files in os.walk(data_dir_train):
    for dir in dirs:
        allfiles[dir] = len(os.listdir(os.path.join(root,dir)))

#mak sure subdirs exists in validation folder
for key, value in allfiles.items():
    dir = os.path.join(data_dir_val,key)
    if not os.path.isdir(dir):
        os.makedirs(dir)

# split it, 80% train, 20% validation
for key, value in allfiles.items():
    dirtrain = os.path.join(data_dir_train, key)
    dirval = os.path.join(data_dir_val, key)

    #get list of files in dir
    files = os.listdir(dirtrain)

    numb_files_to_cpy = int(val_percent * value)

    for i in range(numb_files_to_cpy):
        src = os.path.join(dirtrain,files[i])
        dst = os.path.join(dirval,files[i])
        os.rename(src,dst)

#find max numb images in train (scaled to .8)
biggest = max(allfiles.items(), key=operator.itemgetter(1))[1]*(1-val_percent)


# balance the training dataset
for key, _ in allfiles.items():
    dir = os.path.join(data_dir_train, key)
    files = os.listdir(dir)
    numb_files = len(files)
    numb_to_cpy = int(biggest - len(files))
    if numb_to_cpy == 0:
        continue
    for i in range(numb_to_cpy):
        src = os.path.join(dir, files[i%numb_files])
        suffix = i//numb_files
        dst = os.path.join(dir, "cpy_"+str(suffix) + files[i%numb_files])
        copyfile(src, dst)






