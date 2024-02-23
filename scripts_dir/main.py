import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed

from CNN_model import SimpleCNN
from download_data import Loaders


def run_training(train_loader, val_loader, model, num_epochs=5):

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



if __name__ == '__main__':
    model = SimpleCNN()
    loader  = Loaders(data_path='/home/eliad/github_repo/tech19-home-assigment/data_dir')
    run_training(train_loader=loader.train_loader, val_loader=loader.val_loader, model=model)
    embed()