import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pickle

from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck
from ignite.engine import *
from ignite.metrics import Accuracy, Loss
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

from ignite.handlers import *

results = []
losses = []


class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        # 第一个卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))

def resnet10(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)

def resnet18(**kwargs):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return ResNetCustom(BottleNeck, [3, 4, 6, 3], **kwargs)


def logger(engine, model, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    pbar.log_message(
        "Test Results - Avg accuracy: {:.2f}".format(avg_accuracy)
    )
    results.append(avg_accuracy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256,
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--device', required=False, default=0, type=int,
                        help='CUDA device id for GPU training')
    parser.add_argument('--wd', required=False, default=0.0001, type=float, help='weight decay')
    parser.add_argument('--momentum', required=False, default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma', required=False, default=0.1, type=float, help='gamma')
    options = parser.parse_args()

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    lr = options.lr
    momentum = options.momentum
    wd = options.wd
    gamma = options.gamma
    device = 'cpu' if options.device is None \
        else torch.device('cuda:{}'.format(options.device))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0),
            (63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0))
    ])

    train_set = torchvision.datasets.CIFAR100(root=root, train=True,
                                             download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False,
                                            download=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)


    # define model
    # model = resnet10(num_classes=100)
    model = resnet50(num_classes=100)
    # model = resnet18(num_classes=100)
    model.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0001)
    torch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma = gamma)
    # torch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma = 0.1)
    # scheduler = LRScheduler(torch_lr_scheduler)
    # create ignite engines
    trainer = create_supervised_trainer(model=model,
                                        optimizer=optimizer,
                                        loss_fn=criterion,
                                        device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy()},
                                            device=device)

    # ignite handlers
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=['loss'])

    # trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    
    # print lr at every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_lr():
        print("Learning rate = {:.6f}".format(optimizer.param_groups[0]["lr"]))


    @trainer.on(Events.EPOCH_COMPLETED)
    def print_loss():
        loss = trainer.state.metrics['loss']
        print("Training loss = {:.6f}".format(loss))
        losses.append(loss)
        

    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger, model, evaluator, test_loader, pbar)

    # start training
    t0 = time.time()
    trainer.run(train_loader, max_epochs=epochs)
    t1 = time.time()
    
    print('Best Accuracy:', max(results))
    print('Total time:', t1 - t0)

    with open("./log/losses", "wb") as fp:
        pickle.dump(losses, fp)
    print("Losses successfully written into ./log/losses")

    with open("./log/results", "wb") as fp:
        pickle.dump(results, fp)
    print("Results successfully written into ./log/results")

    # save model
    torch.save(model.state_dict(), './log/model.pth')
    print("Model successfully written into ./log/model.pth")