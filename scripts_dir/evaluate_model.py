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
    print(f'Accuracy of the network on the 10000 test images: {res} %')
    return res


#todo - need to do more generic
def check_different_epoch_numbers(epoch_numbers):
    '''
    
    :param epoch_numbers:
    :return:
    '''
    loader = Loaders(data_path='/home/eliad/github_repo/tech19-home-assigment/data_dir')
    best_res = -1
    for i in epoch_numbers:
        model = SimpleCNN()
        run_training(train_loader=loader.train_loader, val_loader=loader.val_loader, model=model, num_epochs=i)
        print(f'nuber of epochs:{i}')
        res = compare_test_to_model_results(loader.test_loader, model)
        if res > best_res:
            best_res = res
    return best_res


if __name__ == '__main__':
    check_different_epoch_numbers(range(11, 15))
