import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import resnet as RN


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--checkpoint-path', help='Path to checkpoint')
parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
args = parser.parse_args()

model = RN.ResNet18()
if args.ce:
    title = 'Cross entropy'
else:
    title = 'Label smoothing'

base_path = os.path.splitext(args.checkpoint_path)[0]

npy_output_path = f'{base_path}-output.npy'
npy_target_path = f'{base_path}-target.npy'

state_dict = torch.load(args.checkpoint_path)
model.load_state_dict(state_dict)
model.linear = nn.Flatten()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

extract = model
extract.cuda()
extract.eval()

out_target = []
out_output = []

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = extract(inputs)
    output_np = outputs.data.cpu().numpy()
    target_np = targets.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:, np.newaxis])

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)
np.save(npy_output_path, output_array, allow_pickle=False)
np.save(npy_target_path, target_array, allow_pickle=False)

tsne = TSNE(n_components=2, init='pca', random_state=0)
output_array = tsne.fit_transform(output_array)
plt.rcParams['figure.figsize'] = 10, 10
plt.scatter(output_array[:, 0], output_array[:, 1], c=target_array[:, 0])
plt.title(title)
output_path = f'{base_path}-tsne.png'
plt.savefig(output_path, bbox_inches='tight')
