# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:57:20 2021

@author: hlak
"""

import torch
import torchvision
from torch import nn,optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CNN
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset



I3 = np.load(r'\Images.npy')
I4 = np.load(r'\Labels.npy')

I3 = I3.astype('float32')
I4 = I4.astype('float32')




Images2 = np.ones((250, 1, 28, 28))
Images2 = Images2.astype('float32')
temp_3 = 0
temp_4 = 0
for k in range(250) :
    for i in range(28) :
        for j in range(28) :
            temp_3 = j * 5
            Images2[k][0][i][j] = I3[k][temp_4][temp_3]
        temp_4 = (i + 1) * 5
    temp_3 = 0
    temp_4 = 0
I4.resize(250, )

batch_size_test = 200

x_data2=torch.from_numpy(Images2)
y_data2=torch.from_numpy(I4)

test_dataset = TensorDataset(x_data2, y_data2)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size_test,
                         shuffle=True)
'''
def test():
    correct = 0
    total = 0
    print("label       predicted")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("{}          {}".format(int(labels.item()), predicted.data.item()))

        print('CNN trained model： accuracy on my_mnist_dataset set:%d %%' % (100 * correct / total))
'''
cnn = CNN()
cnn.load_state_dict(torch.load(r'D:\study\ml_project\code\test2218\CNN_model_weight2.pth'), False)


eval_loss = 0
eval_acc = 0
criterion = nn.CrossEntropyLoss() 

temp = 0
pred_result = []
for data in test_loader:
    temp = temp+1
    img, label = data
    #Judge whether GPU can be used, and if it can convert data into a format that GPU can process.
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = cnn(img)
    loss = criterion(out,label.long())
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    pre1 = pred.tolist()
    pred_result.append(pre1)
    num_correct = (pred == label).sum()
    accuracy = (pred == label).float().mean()
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc/len(test_dataset)))
#pred_result_1 =  np.stack(int(np.array(pred_result[0])),int(np.array(pred_result[1])))
