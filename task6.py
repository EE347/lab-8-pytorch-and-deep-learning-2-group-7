import time
import cv2

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import functional as transform
import torch.nn.functional as F
import random
import pandas as pd

#       Loss Function  Train Loss  Test Loss  Test Accuracy
# 0  CrossEntropyLoss    0.652985   0.693147            0.5
# 1  CrossEntropyLoss    0.339476   0.693272            0.5
# 2  CrossEntropyLoss    0.348050   0.693373            0.5
# 3  CrossEntropyLoss    0.125078   0.693154            0.5
# 4           NLLLoss    0.648042   0.693182            0.5
# 5           NLLLoss    0.227269   0.693271            0.5
# 6           NLLLoss    0.020275   0.693961            0.5
# 7           NLLLoss    0.007978   0.694922            0.5

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    learning_rate = 0.0005
    batch_size = 16

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Saving parameters
    best_train_loss = 1e9

    # Loss lists
    train_losses = []
    test_losses = []

    results = {
        "Loss Function": [],
        "Train Loss": [],
        "Test Loss": [],
        "Test Accuracy": []
    }


    for loss_type, criterion in [("CrossEntropyLoss", torch.nn.CrossEntropyLoss()), ("NLLLoss", torch.nn.NLLLoss())]:


        # Create the model and optimizer
        model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        # Epoch Loop
        for epoch in range(1, 5):

            # Start timer
            t = time.time_ns()

            # Train the model
            model.train()
            train_loss = 0

            # Batch Loop
            for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

                # Move the data to the device (CPU or GPU)
                images = images.reshape(-1, 3, 64, 64).to(device)
                # labels = labels.reshape(-1, 1).to(device)
                labels = labels.to(device)

                if random.random() > 0.5:
                    images = torch.flip(images, [3])

                angle = random.uniform(-10, 10)
                transform.rotate(images, angle)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Apply LogSoftmax if NLLLoss
                if loss_type == "NLLLoss":
                    outputs = F.log_softmax(outputs, dim=1)


                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Accumulate the loss
                train_loss = train_loss + loss.item()

            # Test the model
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            # Batch Loop
            for images, labels in tqdm(testloader, total=len(testloader), leave=False):

                # Move the data to the device (CPU or GPU)
                images = images.reshape(-1, 3, 64, 64).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Apply LogSoftmax if NLLLoss
                if loss_type == "NLLLoss":
                    outputs = F.log_softmax(outputs, dim=1)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Accumulate the loss
                test_loss = test_loss + loss.item()

                # Get the predicted class from the maximum value in the output-list of class scores
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)

                # Accumulate the number of correct classifications
                correct += (predicted == labels).sum().item()

            test_accuracy = correct/total
            train_loss_avg = train_loss / len(trainloader)
            test_loss_avg = test_loss / len(testloader)

            results["Loss Function"].append(loss_type)
            results["Train Loss"].append(train_loss_avg)
            results["Test Loss"].append(test_loss_avg)
            results["Test Accuracy"].append(test_accuracy)
            # Print the epoch statistics
            print(f'Epoch: {epoch}, Loss Type: {loss_type}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

            # Update loss lists
            train_losses.append(train_loss_avg)
            test_losses.append(test_loss_avg)

            # Update the best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), './best_model.pth')

            # Save the model
            torch.save(model.state_dict(), './current_model.pth')



            # Create the loss plot
            plt.plot(train_losses, label=f'Train Loss ({loss_type})')
            plt.plot(test_losses, label=f'Test Loss ({loss_type})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('./task5_loss_plot.png')

            results_df = pd.DataFrame(results)
            print(results_df)