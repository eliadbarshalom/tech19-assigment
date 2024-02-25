import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CNN_model import SimpleCNN
from download_data import Loaders


def run_training(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, num_epochs=5,
                 save_model_file=None):
    '''
    This function run a model training on a chosen data set
                                and changes the model it gets.
    :param train_loader: the train set for the model
    :param val_loader: the validation set for the model
    :param model: a pytorch module to train on the data
    :param num_epochs: number of epochs for the model training
    '''
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        # Validate the model
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('[%d] val loss: %.3f, val accuracy: %.3f %%' %
              (epoch + 1, val_loss / len(val_loader), 100 * correct / total))
    print('Finished Training')
    if save_model_file is not None:
        print(f"Saving the model to {save_model_file}")
        torch.save(model.state_dict(), save_model_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, help='the path of the directory of the data sets files')
    parser.add_argument('--model_file_path', type=str, help='the path of the file to save the model')
    parser.add_argument('--epoch_number', type=int, help='The number of training loops to do')
    args = parser.parse_args()

    model = SimpleCNN()
    loader = Loaders(data_path=args.data_path)
    run_training(train_loader=loader.train_loader, val_loader=loader.val_loader, model=model,num_epochs=args.epoch_number,
                 save_model_file=args.model_file_path)

