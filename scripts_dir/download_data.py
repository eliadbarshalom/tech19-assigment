import os

import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
import pickle



class Loaders(object):
    '''
    This object holds the DataLoders of all the data sets for a model training and evaluation
    '''
    def __init__(self, train_loader=None, val_loader=None, test_loader=None, data_path=None):
        '''
        :param train_loader: torch DataLoder object of the train set
        :param val_loader: torch DataLoder object of the validation set
        :param test_loader: torch DataLoder object of the test set (after training evaluation)
        :param data_path: there is an option to initialize this object with path of a directory holding all the data files
        '''
        assert None not in [train_loader, val_loader, test_loader] or data_path is not None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if data_path is not None:
            self.load_data(data_path=data_path)

    def load_data(self, data_path):
        '''
        This function load all relevant data from files  in a given directory
        :param data_path: the path of the given directory
        '''
        #todo: make this function prettier
        if self.train_loader is None:
            self.train_loader = self.load_data_file(os.path.join(data_path, 'train_loader.pkl'))
        if self.val_loader is None:
            self.val_loader = self.load_data_file(os.path.join(data_path, 'val_loader.pkl'))
        if self.test_loader is None:
            self.test_loader = self.load_data_file(os.path.join(data_path, 'test_loader.pkl'))


    @staticmethod
    def load_data_file(file_full_path):
        '''
        This function loads a given DataLoader file
        :param file_full_path: the path of a given file
        :return: torch DataLoader object with the wanted data
        '''
        #todo - if doesnt exist can run the download function
        assert os.path.exists(file_full_path)
        with open(file_full_path, 'rb') as f:
            return pickle.load(f)


def download_CIFAR10_data():
    '''
    This function download the files of CIFAR10 if there not downloaded allready, does some manipulation on the data, creates DataLoaders objects of train, validation and test sets and save them to files
    :return: Loaders object with the sets
    '''
    # Define transformations to apply to the dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image within -10 to +10 degrees
        transforms.RandomResizedCrop(32),  # Randomly crop and resize the image to 32x32
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert PIL image or numpy.ndarray to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image data
    ])
    # Download CIFAR-10 training dataset
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Download CIFAR-10 test dataset
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    #compute the amount of indexes for a validation set
    #the 0.9 is as thumb rule general estimation and can be changed by tests
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size

    #split a validation set from the downloaded train set,
    # when making sure tere will be an equale represantion for each label
    label_to_indexes_pool = defaultdict(list)
    for i, (_, label) in enumerate(full_train_set):
        label_to_indexes_pool[label].append(i)
    val_indexes = [x for k in label_to_indexes_pool for x in label_to_indexes_pool[k][:int(val_size/len(label_to_indexes_pool))]]
    train_indexes = list(set(range(train_size)) - set(val_indexes))

    val_subset = Subset(full_train_set, val_indexes)
    train_subset = Subset(full_train_set, train_indexes)


    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


    # Save train_loader to a file
    with open('train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)

    # Save val_loader to a file
    with open('val_loader.pkl', 'wb') as f:
        pickle.dump(val_loader, f)

    # Save test loader to a file
    with open('test_loader.pkl', 'wb') as f:
        pickle.dump(test_loader, f)

    return Loaders(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)




if __name__ == '__main__':
    #todo argparse on run modes
    download_CIFAR10_data()
