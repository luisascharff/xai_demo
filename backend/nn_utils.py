import torch 
import torch.nn as nn 
from torchvision import datasets, transforms
from zennit.composites import EpsilonPlus
from crp.attribution import CondAttribution

# Define dataset, loaders, and transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform) 
testset = datasets.MNIST('./data', download=True, train=False, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) 
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True) 

class Net(nn.Module):
    def __init__(self, n=16):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(784, n)
        self.flatten = nn.Flatten()
        self.finale_layer = nn.Linear(n, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layer(x)
        x = self.finale_layer(x)
        return x

def train_mnist(model, trainloader, testloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

def generate_crp_heatmap(model, idx, hidden_neuron_idx, output_neuron_idx):
    vis_testset = datasets.MNIST('./data', 
                                 download=True, 
                                 train=False, 
                                 transform=transforms.Compose([transforms.ToTensor()])) 
    attributor = CondAttribution(model, EpsilonPlus()) 
    conditions = [{
        'y': output_neuron_idx,
        'hidden_layer': hidden_neuron_idx
    }]
    
    sample = testset[idx][0]
    sample = sample.unsqueeze(0)    
    sample.requires_grad_()
    
    attribution = attributor(sample, conditions=conditions)
    heatmap = attribution.heatmap
    return heatmap.numpy()

# Train the model upon module import
model = train_mnist(Net(16), trainloader, testloader)