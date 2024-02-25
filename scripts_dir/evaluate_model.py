import argparse
import os.path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from CNN_model import SimpleCNN
from download_data import Loaders
from main import run_training

def compare_test_to_model_results(test_loader:DataLoader, model:nn.Module):
    '''
    This functions gives the model results by testing it on a test set
    :param test_loader: the test set for testing the model
    :param model: the tested model
    :return:
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    res = int(100 * correct / total)

    print(f'Accuracy of the network : {res} %')



def check_different_epoch_numbers(epoch_numbers, loader):
    '''
    This function is using for comparing different number of epochs for te model training
    :param epoch_numbers:array of epoch numbers to check
    :param loader: the loader to et te trest set from
    :return:the model results on the loader test with the specific epoch number
    '''
    best_res = -1
    for i in epoch_numbers:
        model = SimpleCNN()
        run_training(train_loader=loader.train_loader, val_loader=loader.val_loader, model=model, num_epochs=i)
        print(f'number of epochs:{i}')
        res = compare_test_to_model_results(loader.test_loader, model)
        if res > best_res:
            best_res = res
    return best_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path of the directory of the data sets files')
    parser.add_argument('--model_file_path', type=str, help='the path of the file to load the model from')
    args = parser.parse_args()
    loader = Loaders(data_path=args.data_path)
    model = SimpleCNN().get_trained_model_from_file(args.model_file_path)
    compare_test_to_model_results(loader.test_loader, model=model)
