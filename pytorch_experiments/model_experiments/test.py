import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

from PIL import Image
from time import time

from models.lenet import LeNet
from models.noisy_lenet import NoisyLeNet


data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

preprocess = transforms.Compose([
   transforms.Resize((32, 32)),
   transforms.ToTensor()
])

criterion = nn.CrossEntropyLoss()

def test_on_data():
    net = torch.load('trained-lenet.pt')
    print(net)

    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        inputs, truth = Variable(images), Variable(labels)
        outputs = net(inputs)
        avg_loss += criterion(outputs, truth).sum()
        preds = outputs.data.max(1)[1]
        total_correct += preds.eq(truth.data.view_as(preds)).sum()
    
    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.item(), float(total_correct) / len(data_test)))


    noisy_net = NoisyLeNet(0.001)
    noisy_net.load_state_dict(net.state_dict())

    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        inputs, truth = Variable(images), Variable(labels)
        outputs = torch.zeros([inputs.size()[0], 10])
        for k in range(len(inputs)):
            outputs[k] = noisy_net(inputs[k][None, :, :, :])
        avg_loss += criterion(outputs, truth).sum()
        preds = outputs.data.max(1)[1]
        total_correct += preds.eq(truth.data.view_as(preds)).sum()
    
    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.item(), float(total_correct) / len(data_test)))

def forward_image():
    net = torch.load('trained-lenet.pt')
    print(net)
    
    image = Image.open('inputs/3.png').convert('L')
    input = preprocess(image).unsqueeze(0)
    print(input.size())

    input = 1 - input

    output = net(input)
    print (output.argmax())

    print (output)

def test_logic_faults():
    net = torch.load('trained-lenet.pt')

    #noisy_net = NoisyLeNet(0.001)
    noisy_net = NoisyLeNet(0.1)
    noisy_net.load_state_dict(net.state_dict())
    #noisy_net = noisy_net.float()
    
    image = Image.open('inputs/4.png').convert('L')
    input = preprocess(image).unsqueeze(0)
    print(input.size())
    input = 1 - input

    start = time()
    outputb = net(input)
    end = time()
    print (outputb.argmax(), 'computed in', (end-start), 'secondes.')
    #print (outputb.double())

    start = time()
    outputa = noisy_net(input)
    end = time()
    print (outputa.argmax(), 'computed in', (end-start), 'secondes.')
    #print (outputa.double())

    #print(outputa.float() - outputb)


if __name__ == '__main__':
    test_on_data()
