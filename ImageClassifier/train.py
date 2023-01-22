import argparse
from time import time
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models


# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str, 
        help = 'path to the folder of train images')
parser.add_argument('--save_dir', type = str, 
        help = 'saving model to specifics directory: /path/to/model.pth', default = './saved_model.pth' )
parser.add_argument('--arch', default = 'resnet50' )
parser.add_argument('--learning_rate', default = 0.001, type = float )
parser.add_argument('--hidden_units', default = 0, type = int )
parser.add_argument('--epochs', default = 15, type = int )
parser.add_argument('--gpu', action="store_true", default = False )
in_args = parser.parse_args()


def main():
    trainloader, testloader, validloader, num_cls, class_to_idx = preprocess_img(in_args.data_dir)

    model_arch = Model_Cls(in_args.arch, num_cls, in_args.learning_rate, in_args.hidden_units)
    Train_model(trainloader, testloader, validloader, model_arch, in_args.epochs, class_to_idx)
    

def preprocess_img(data_dir):
    # define dataset directory 
    base_path = os.listdir(data_dir)
    train_path = os.path.join(data_dir, base_path[0])
    test_path = os.path.join(data_dir, base_path[1])
    val_path = os.path.join(data_dir, base_path[2])
    
    num_cls = len(os.listdir(train_path))
    
    # define transforms
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=255, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=25),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(size=255),
        transforms.RandomResizedCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    train_datasets = datasets.ImageFolder(train_path, transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    
    test_datasets = datasets.ImageFolder(test_path, transform=valid_transforms)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    
    val_datasets = datasets.ImageFolder(val_path, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(val_datasets, batch_size=32, shuffle=True)

    class_to_idx = train_datasets.class_to_idx.items()
    
    return trainloader, testloader, validloader, num_cls, class_to_idx

            
def Hidden_fc(num_cls, num_hiddens):
    if(num_hiddens > 0):
        model_classifier = lambda input_node: nn.Sequential(
            nn.Linear(input_node, num_hiddens),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hiddens, num_cls),
            nn.LogSoftmax(dim=1)
        )
    else:
        model_classifier = lambda input_node: nn.Sequential(
            nn.Linear(input_node, num_cls),
            nn.LogSoftmax(dim=1)
        )
    return model_classifier


def Model_Cls(arch, num_cls, learning_rate, hiddens):
    model_classifier = Hidden_fc(num_cls, hiddens)
    
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        for param in model.parameters():
            param.requires_grad = False
        model.fc = model_classifier(num_ftrs)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        num_ftrs = model.classifier[0].in_features
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = model_classifier(num_ftrs)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[0].in_features
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = model_classifier(num_ftrs)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    else:
        print(arch+" model unknow")
        exit()
   
    return model, criterion, optimizer


def toDevice(images, labels, model):  # move tensors to GPU if CUDA is available
    if in_args.gpu == True:
        if torch.cuda.is_available() == False:
            images, labels = images.to("cpu"), labels.to("cpu")
            model = model.to("cpu")
            print("GPU not found!!")
            print("Train using CPU..")
        else:
            images, labels = images.to("cuda:0"), labels.to("cuda:0")
            model = model.to("cuda:0")
    elif in_args.gpu == False:
        images, labels = images.to("cpu"), labels.to("cpu")
        model = model.to("cpu")
    return images, labels, model
        

def Train_model(trainloader, testloader, validloader, model_arch, n_epochs, class_to_idx):
    model, criterion, optimizer = model_arch
    
    train_losses, valid_losses, valid_accuracies = [], [], []
    start_time = time()
    for e in range(n_epochs):
        tot_train_loss = 0
        for images, labels in trainloader:   
            images, labels, model = toDevice(images, labels, model)
            
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            tot_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        else:
            tot_valid_loss, valid_correct = Validation_testing(validloader, model, criterion)
            model.train()

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(trainloader)
            valid_loss = tot_valid_loss / len(validloader)
            valid_acc = valid_correct / len(validloader)

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)

            print("Epoch: {}/{}.. ".format(e+1, n_epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "validation Loss: {:.3f}.. ".format(valid_loss),
                  "validation Accuracy: {:.3f}".format(valid_acc))
    tot_time =  time() - start_time
    
    print("===== Training completed =====")
    print(f"total duration during trainig: {100* (tot_time/60):.2f} minutes")
    print()
    
    _, test_correct = Validation_testing(testloader, model)
    print(f"Accuracy of the network on the test images: {test_correct / len(testloader) * 100}")
    print(test_correct / len(testloader))

    SaveModel(model, optimizer, train_losses, valid_losses, valid_accuracies, class_to_idx)


def Validation_testing(test_loader, model, criterion=None):
    tot_loss = 0
    num_correct = 0  # Number of correct predictions on the test set
    # Turn off gradients for numation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels, model = toDevice(images, labels, model)
            output = model(images)
            if criterion is not None:
                loss = criterion(output, labels)
                tot_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            num_correct += equals.sum().item()
    return tot_loss, num_correct


def SaveModel(model, optimizer, train_losses, valid_losses, valid_accuracies, class_to_idx):
    head_tail = os.path.split(in_args.save_dir) # split file path
    if not os.path.exists(head_tail[0]):  # crete folder if save_dir path doesn't exist
        os.makedirs(head_tail[0])
        
    checkpoint = {
        'train_loss' : train_losses,
        'val_loss' : valid_losses,
        'val_acc' : valid_accuracies,
        'epochs' : in_args.epochs,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'idx_to_class': {v: k for k, v in class_to_idx}
    }
    torch.save(checkpoint, in_args.save_dir)


if __name__ == "__main__":
    main()