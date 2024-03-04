**Tech19-assigment**

**Description**

This project is build of all levels of the CIFAR10 model. 
Download the data, training the model and evaluationg the model.
All thesteps are written here and some of the results

**Getting Started**

To run this project, follow the instructions below.


**Prerequisites**

 python>=3.9

Installation
pip install the added Pipfile

**Usage**

In order to test an existing model on the uploaed test set:
class name can be any of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```
python ./tech19-assigment/scripts_dir/evaluate_model.py --data_path tech19-assigment/data_dir --model_file_path tech19-assigment/data_dir/model_final.pkl --class_name <class from the list>
```

In order to show model results on random images from the test set (The prediction will be printed in the terminal and images grid will pop):
```
python ./tech19-assigment/scripts_dir/img_show.py --data_path tech19-assigment/data_dir --model_file_path tech19-assigment/data_dir/model_final.pkl
```

In case you want to download the train and validation data into the data_dir directory:
```
python ./tech19-assigment/scripts_dir/download_data.py
```

In case you want to train the model on the data set you first need the above download and the run:
```
python ./tech19-assigment/scripts_dir/main.py --data_path tech19-assigment/data_dir --model_file_path <path to save the model file in > --epoch_number <number of epochs for model training>
```
