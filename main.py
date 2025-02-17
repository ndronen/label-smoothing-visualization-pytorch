'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import resnet as RN
from utils import progress_bar, LabelSmoothingCrossEntropy, save_model


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--output-dir', type=str, required=True,
    help='save checkpoints to directory'
)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument(
    '--resume', '-r', action='store_true', help='resume from checkpoint'
)
parser.add_argument('--ce', action='store_true', help='Cross entropy use')
parser.add_argument('--load-checkpoint', type=str, help='checkpoint path')
parser.add_argument(
    '--save-best-only', action='store_true', help='only save best models'
)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=8
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
)

# Model
print('==> Building model..')
net = RN.ResNet18()
if args.load_checkpoint:
    state_dict = torch.load(args.load_checkpoint)
    net.load_state_dict(state_dict)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

if args.ce:
    criterion = nn.CrossEntropyLoss()
    print("Use CrossEntropy")
else:
    criterion = LabelSmoothingCrossEntropy()
    print("Use Label Smoothing")


if args.load_checkpoint:
    # Assume we're fine-tuning
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr * 0.1**3,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1
    )
else:
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90]
    )


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss/(batch_idx+1), 100.*correct/total, correct, total
            )
        )
    scheduler.step()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (  # noqa: E501
                    test_loss/(batch_idx+1), 100.*correct/total, correct, total
                )
            )

    # Save checkpoint.
    acc = 100.*correct/total
    if not args.save_best_only:
        save_path = f'{args.output_dir}/epoch-{epoch:03d}.bin'
        save_model(net, save_path)
    if acc > best_acc:
        # Always save the best model.
        save_best_path = f'{args.output_dir/best.bin'
        print(f'Saving new best model')
        save_model(net, save_best_path)
    best_acc = acc


for epoch in range(start_epoch, start_epoch+120):
    train(epoch)
    test(epoch)
