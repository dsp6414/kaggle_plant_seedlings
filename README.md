# kaggle_plant_seedlings
Determine the species of a plant seedling based on a picture.  Uses PyTorch and Resnet.<br>
See [Kaggle Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) for details and datasets

## Run python files in this order
* strip_image_backgroung.py - uses openCV to mask out irrelevant background in images
* split_dataset.py - splits train set into train and validate folders.Balances dataset by making copies of minority images
* train.py - finetunes a Resnet50 network in 4 phases, generates a predictions file on test data after each phase
  * 1st trains just the FC layer, all other layers frozen
  * 2nd trains the whole network with differential learning rates applied to each layer
  * 3rd trains using all the train and validation data (increases training data), uses custom Pytorch DataSet
  * 4th trains using all the data from 3 plus knowledge distillation (test data and 3rd phase predictions on test data, increases training data by asssumming predictions are mostly correct) 

## Other files 
utils.py - contains custom DataSet classes
* CSV_Dataset - Generates a dataset from a csv file of the form

    file,species<br>
    25cf6eb73.png,Maize<br>
    953496deb.png,Fat Hen<br>

    where the first param is the name of the file, the second is the prediction<br>
    This class is used for the knowledge distillation portion of the training cycle
* Unlabeled_Dataset - provides access to unlabled files in a directory for testing

## Accuracy
* pred_1.csv - 82.367%
* pred_2.csv - 90.302%
* pred_3.csv - 95.591%
* pred_4.csv - 95.465%  #does not look like this step helped much

