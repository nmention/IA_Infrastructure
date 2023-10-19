#!/usr/bin/env python
# coding: utf-8

# <img src='https://upload.wikimedia.org/wikipedia/fr/thumb/e/ed/Logo_Universit%C3%A9_du_Maine.svg/1280px-Logo_Universit%C3%A9_du_Maine.svg.png' width="300" height="500">
# <br>
# <div style="border: solid 3px #000;">
#     <h1 style="text-align: center; color:#000; font-family:Georgia; font-size:26px;">Infrastructures pour l'IA</h1>
#     <p style='text-align: center;'>Master Informatique 1</p>
#     <p style='text-align: center;'>Anhony Larcher</p>
# </div>

# Dans cet exercice nous allons classifier les image sde la base de données MNIST un réseau de neurones profonds

# ## Importez un package qui vous permette de tracer ces graphiques

# In[9]:


#todo...

import matplotlib.pyplot as plt


# ## Importez PyTorch et téléchargez MNIST

# In[10]:


import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])




# Get the test dataset
test_set = datasets.MNIST(root = './',
                          download = True, train = False, transform = transform)


# # Préparez les données d'apprentissage et de développement

# In[11]:


batch_size = 16

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST('./files/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
                                       
                                       
validation_set = torchvision.datasets.MNIST('./files/', train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)


# # Regardez quelques données

# In[12]:


examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape


# ## Visualisez quelques exemples de MNIST
# Chaque ligne de X est un vecteur qui contient les 784 valeurs des pixels d'une image 28x28 en niveaux de gris
# 
# Affichez les 4 premiers exemples de cette base de données avec leur label

# In[13]:


fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig


# # Créez un réseau de neurones

# In[14]:


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # Second 2D convolutional layer, taking in the 32 input layers,
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # First fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output
    
    
network = Net()
print(network)


# # Testez votre réseau en passant des données dedans

# In[15]:


# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))

network = Net()
result = network(random_data)
print (result)


# # Créez un optimizer

# In[16]:


lr = 1e-3
momentum = 0.5
log_interval = 10
n_epochs = 10
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)


# # Entrainez votre réseau

# In[17]:


train_losses = []
train_counter = []
validation_losses = []
validation_counter = [i*len(validation_loader.dataset) for i in range(n_epochs + 1)]


# In[18]:


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')


# In[19]:


def validation():
  network.eval()
  validation_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in validation_loader:
      output = network(data)
      validation_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  validation_loss /= len(validation_loader.dataset)
  validation_losses.append(validation_loss)
  print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    validation_loss, correct, len(validation_loader.dataset),
    100. * correct / len(validation_loader.dataset)))


# # Boucle d'apprentissage
# 
# On teste une première fois le modèle avec des paramètres aléatoires avant l'apprentissage

# In[21]:


# !mkdir results
validation()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  validation()


# # Tracez les performances d'apprentissage

# In[ ]:


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(validation_counter, validation_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig


# # Passez sur GPU
# Modifiez maintenant le code pour l'exécuter sur GPU et comparez la vitesse d'exécution

# In[ ]:





# In[ ]:


# net = network()
# net = net.to('cuda')
# for (data,target in ())

# output = net(data.to('cuda'))

# loss = loss(output,target.to('cuda'))
# loss.item()

