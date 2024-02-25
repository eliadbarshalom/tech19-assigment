import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
from CNN_model import SimpleCNN
from download_data import Loaders


def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_model_results_on_test(model:torch.nn, data_set:DataLoader):

    random_indices = torch.randperm(len(data_set.test_loader.dataset)).tolist()[:10]
    random_sampler = torch.utils.data.SubsetRandomSampler(random_indices)

    # Create a new DataLoader with the random sampler
    random_dataloader = DataLoader(data_set.test_loader.dataset, batch_size=10, sampler=random_sampler)
    for batch in random_dataloader:
        images, labels  = batch
        break

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    print(','.join([data_set.test_loader.dataset.classes[i] for i in predicted]))
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path of the directory of the test set file')
    parser.add_argument('--model_file_path', type=str, help='the path of the file to load the model from')
    args = parser.parse_args()
    loader = Loaders(data_path=args.data_path, only_test=True)
    model = SimpleCNN().get_trained_model_from_file(args.model_file_path)
    show_model_results_on_test(model=model, data_set=loader)

