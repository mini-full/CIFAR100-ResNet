import time
from typing import Callable, List, Optional, Type, Union
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pickle

from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck, resnet50, resnet18
from ignite.engine import *
from ignite.metrics import Accuracy
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

from ignite.handlers import *

results = []
losses = []


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

    parser.add_argument('--root', required=False, type=str, default='./Traffic_sign',
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
    device = 'cpu'

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            # (125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0),
            # (63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0))
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load dataset

    # data set structure, all the files are *.jpg
    # - Traffic_sign
    #     - train_dataset
    #         - GuideSign
    #         - M1
    #         - M4
    #         - M5
    #         - M6
    #         - M7
    #         - P1
    #         - P10_50
    #         - P12
    #         - W1
    
    #     - test_dataset
    #         - GuideSign
    #         - M1
    #         - M4
    #         - M5
    #         - M6
    #         - M7
    #         - P1
    #         - P10_50
    #         - P12
    #         - W1

    train_set = torchvision.datasets.ImageFolder(root=root + '/train_dataset', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize, shuffle=True, num_workers=workers)
    test_set = torchvision.datasets.ImageFolder(root=root + '/test_dataset', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize, shuffle=False, num_workers=workers)


    # define model
    # model = resnet10(num_classes=10)
    # model = resnet50(num_classes=10)
    model = resnet18(num_classes=10)
    # model = torchvision.models.resnet50(num_classes=100, pretrained=True)

    # read model from file
    # model.load_state_dict(torch.load('./load/model.pth'))
    # modules = model.modules()
    # for p in modules:
    #     if p._get_name()!= 'Linear':
    #         p.requires_grad = False
    model.to(device)
            

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0001)
    torch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma = 0.1)
    scheduler = LRScheduler(torch_lr_scheduler)
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

    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    
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