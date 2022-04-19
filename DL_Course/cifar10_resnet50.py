# 笔记本显存不够，需要分批训练...

import torch
from torch.utils.tensorboard.summary import image
import torchvision
# import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

myWriter = SummaryWriter('./tensorboard/log/')

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])    # 均值标准差和上一个有出入

#  load
train_dataset = torchvision.datasets.CIFAR10(root=r'./Datas/CIFAR10', train=True, download=True,
                                             transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root=r'./Datas/CIFAR10', train=False, download=True,
                                            transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# 定义模型
# myModel = torchvision.models.resnet50(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
myModel = torchvision.models.resnet18(pretrained=True)      # 50太慢，而且显存不够，2个epoch--1h(测试准确率94%左右)
# resnet18显存可以，但是显卡很吓人
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

# 损失函数及优化器
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

learning_rate = 0.005
myOptimzier = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(10):
    training_loss = 0.0
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  # GPU加速
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step=_epoch * len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0:
            print('[iteration - %3d] training loss: %.3f' % (_epoch * len(train_loader) + _step, training_loss / 10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0
    torch.save(myModel, './Mudule/Resnet18_Own.pkl') # 保存整个模型
    'pkl,pt,pth等格式的pytorch模型，除了名字不同，实际上是一样的，编码格式相同'

    myModel.eval()      # 测试
    for images, labels in test_loader:
        # GPU加速
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Testing Accuracy : %.3f %%' % (100 * correct / total))
    myWriter.add_scalar('test_Accuracy', 100 * correct / total)
