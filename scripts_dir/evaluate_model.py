import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from CNN_model import SimpleCNN
from download_data import Loaders
from main import run_training
from tabulate import tabulate


def get_model_accuracy(test_loader:DataLoader, model:nn.Module):
    '''
    This functions gives the model accuracy by testing it on a test set
    true positive/ total
    :param test_loader: the test set for testing the model
    :param model: the tested model
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
    return float(int((correct / total)*100)/100)



def get_model_precision_for_one_class(class_name,test_loader:DataLoader, model:nn.Module):
    '''
    This functions gives the model precision acording to a given class by testing it on a test set
    true positive / true positive + false positive
    :param test_loader: the test set for testing the model
    :param model: the tested model
    '''
    TP = 0
    total = 0
    class_name_label = test_loader.dataset.classes.index(class_name)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            TP += ((predicted == labels) & (predicted == class_name_label)).sum().item()
            total += (predicted == class_name_label).sum().item()

    return float(int((TP / total)*100)/100)


def get_model_recall_for_one_class(class_name, test_loader: DataLoader, model: nn.Module):
    '''
    This functions gives the model recall acording to a given class by testing it on a test set
    true positive / true positive + false negative
    :param test_loader: the test set for testing the model
    :param model: the tested model
    '''
    TP = 0
    total = 0
    class_name_label = test_loader.dataset.classes.index(class_name)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            TP += ((predicted == labels) & (predicted == class_name_label)).sum().item()
            total += (labels == class_name_label).sum().item()

    return float(int((TP / total)*100)/100)


def get_model_F1Score_for_one_class(class_name, test_loader: DataLoader, model: nn.Module):
    '''
    This functions gives the model F1 score acording to a given class by testing it on a test set
    true 2* ((precision*recall)/(precision+recall))
    :param test_loader: the test set for testing the model
    :param model: the tested model
    '''
    precision = get_model_precision_for_one_class(class_name=class_name, test_loader=test_loader, model=model)
    recall = get_model_recall_for_one_class(class_name=class_name, test_loader=test_loader, model=model)
    return float(int(200 * ((precision*recall)/(precision+recall)))/100)


def print_model_results_on_test_set(class_name, test_loader: DataLoader, model: nn.Module):

    table_data = [
    [class_name, "results"],
    ["Accuracy", get_model_accuracy(test_loader=test_loader, model=model)],
    ["Precision", get_model_precision_for_one_class(class_name=class_name, test_loader=test_loader, model=model)],
    ["Recall", get_model_recall_for_one_class(class_name=class_name, test_loader=test_loader, model=model)],
    ["F1-Score", get_model_F1Score_for_one_class(class_name=class_name, test_loader=test_loader, model=model)]
    ]
    # Print the table
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

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
        res = get_model_accuracy(loader.test_loader, model)
        if res > best_res:
            best_res = res
    return best_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path of the directory of the data sets files')
    parser.add_argument('--model_file_path', type=str, help='the path of the file to load the model from')
    parser.add_argument('--class_name', type=str, help='class name to precision, recall and f1-score')
    args = parser.parse_args()
    loader = Loaders(data_path=args.data_path, only_test=True)
    model = SimpleCNN().get_trained_model_from_file(args.model_file_path)
    print_model_results_on_test_set(class_name=args.class_name, test_loader=loader.test_loader, model=model)
