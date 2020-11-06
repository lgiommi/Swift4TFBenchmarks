import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt
#import copy

def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 40),
        nn.ReLU(),
        nn.Linear(40, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model

def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        #images = images.cuda()
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total

def train(numb_epoch=3, lr=0.1, device="cpu"):
    accuracies_train = []
    accuracies_val = []
    cnn = create_lenet().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=lr)
    max_accuracy = 0
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy_train = float(validate(cnn, train_dl))
        accuracy_val = float(validate(cnn, val_dl))
        accuracies_train.append(accuracy_train)
        accuracies_val.append(accuracy_val)
        print(f"Epoch {epoch+1};\taccuracy_train: {accuracy_train};\taccuracy_val: {accuracy_val}")
        #if accuracy_val > max_accuracy:
        #    best_model = copy.deepcopy(cnn)
        #    max_accuracy = accuracy_val
        #    print("Saving Best Model with Accuracy: ", accuracy)
        #print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    #plt.plot(accuracies_train)
    #return best_model

numb_batch = 128

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

train_dl = torch.utils.data.DataLoader(train_data, batch_size = numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = numb_batch)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

lenet = train(5, device=device)
