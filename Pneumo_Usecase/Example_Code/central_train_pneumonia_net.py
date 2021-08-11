import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix

data_dir = '../chest_xray'
random_seed = 2020

np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

train_dataset = ImageFolder(data_dir+'/train', transform=tt.Compose(
    [tt.Resize(255),
     tt.CenterCrop(224),
     tt.RandomHorizontalFlip(),
     tt.RandomRotation(10),
     tt.RandomGrayscale(),
     tt.RandomAffine(translate=(0.05,0.05), degrees=0),
     tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ,inplace=True),
     tt.ToTensor()
    ]))

train_size = round(len(train_dataset)*0.7) # 70%
val_size = len(train_dataset) - train_size # 30%
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)  # yield will stop here, perform other steps, and the resumes to the next loop/batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

def F1_score(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    # precision, recall, and F1
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1, preds

class PneumoniaModelBase(nn.Module):

    # this is for loading the batch of train image and outputting its loss, accuracy
    # & predictions
    def training_step(self, batch, weight):
        images, labels = batch
        out = self(images)  # generate predictions
        loss = F.cross_entropy(out, labels, weight=weight)  # weighted compute loss
        acc, preds = accuracy(out, labels)  # calculate accuracy

        return {'train_loss': loss, 'train_acc': acc}

    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]  # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['train_acc'] for x in outputs]  # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}

    # this is for loading the batch of val/test image and outputting its loss, accuracy,
    # predictions & labels
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # generate predictions
        loss = F.cross_entropy(out, labels)  # compute loss
        acc, preds = accuracy(out, labels)  # calculate acc & get preds

        return {'val_loss': loss.detach(), 'val_acc': acc.detach(),
                'preds': preds.detach(), 'labels': labels.detach()}

    # detach extracts only the needed number, or other numbers will crowd memory

    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]  # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['val_acc'] for x in outputs]  # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
              format(epoch + 1, train_result['train_loss'], train_result['train_acc'],
                     val_result['val_loss'], val_result['val_acc']))

    # this is for using on the test set, it outputs the average loss and acc,
    # and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies
        # combine predictions
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]
        # combine labels
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels}

class PneumoniaResnet(PneumoniaModelBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Freeze training for all layers before classifier
        for param in self.network.fc.parameters():
            param.require_grad = False
        num_features = self.network.fc.in_features  # get number of in features of last layer
        self.network.fc = nn.Linear(num_features, 2)  # replace model classifier

    def forward(self, xb):
        return self.network(xb)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, lr, model, train_loader, val_loader, weight, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()  # release all the GPU memory cache
    history = {}
    optimizer = opt_func(model.parameters(), lr)
    best_loss = 1  # initialize best loss, which will be replaced with lower better loss
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_outputs = []
        lrs = []
        for batch in train_loader:
            outputs = model.training_step(batch, weight)
            loss = outputs['train_loss']  # get the loss
            train_outputs.append(outputs)
            train_results = model.train_epoch_end(train_outputs)
            loss.backward()  # compute gradients
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()  # update weights
            optimizer.zero_grad()  # reset gradients

        # Validation phase
        val_results = evaluate(model, val_loader)
        # Save best loss
        if val_results['val_loss'] < best_loss and epoch + 1 > 15:
            best_loss = min(best_loss, val_results['val_loss'])
            best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model.state_dict(), 'best_model.pt')
        # print results
        model.epoch_end(epoch, train_results, val_results)
        # save results to dictionary
        to_add = {'train_loss': train_results['train_loss'],
                  'train_acc': train_results['train_acc'],
                  'val_loss': val_results['val_loss'],
                  'val_acc': val_results['val_acc'], 'lrs': lrs}
        # update performance dictionary
        for key, val in to_add.items():
            if key in history:
                history[key].append(val)
            else:
                history[key] = [val]
    model.load_state_dict(best_model_wts)  # load best model
    return history, optimizer, best_loss

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = to_device(PneumoniaResnet(), device)

epochs = 20
lr = 0.0001
grad_clip = None
weight_decay = 1e-4
opt_func = torch.optim.Adam
# weighted loss for data class imbalance
weight = torch.FloatTensor([3876/(1342+3876), 1342/(1342+3876)]).to(device)

history, optimizer, best_loss = fit(epochs, lr, model, train_dl, val_dl, weight, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)