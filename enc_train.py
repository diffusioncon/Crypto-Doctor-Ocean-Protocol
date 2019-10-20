import logging
import os
import shutil
import time
from random import shuffle

import crypten
import crypten.communicator as comm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from examples.meters import AverageMeter
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder


def get_img_files(path: str):
    files = os.listdir(path)
    return [file for file in files if file.endswith('.jpg')]


class BrainSet(Dataset):
    """
    Provides non-transformed, random access to tuples of OpenCV images and
      entities.
    """

    def __init__(self, image_files, root: str, transform):
        """
        Args:
            image_files: Images specified as URLs, local files or references that
              the optional FileManager can open in binary.
            entity_lists: List of training data.
            file_manager: Optionally load images through this FileManager
        """

        self.image_files = image_files
        self.root = root

        self.transform = transform

    def get_image(self, index):
        fn = self.image_files[index]
        return Image.open(os.path.join(self.root, fn)).convert('RGB')

    def get_gt(self, index):
        fn = self.image_files[index]
        if 'Y' in fn:
            return 1
        else:
            return 0

    def __getitem__(self, index):
        return self.transform(self.get_image(index)), torch.from_numpy(np.array(self.get_gt(index)))

    def __len__(self):
        return len(self.image_files)


def get_model(model_name: str, num_classes: int):
    model = getattr(models, model_name)(pretrained=True)
    in_features = model._modules['fc'].in_features
    model._modules['fc'] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    return model


# def get_model(model_name, num_classes):
#   return LeNet()


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # init crypten
    crypten.init()

    # setup
    path = '/Users/lukassanner/Downloads/brain-mri-images-for-brain-tumor-detection/'
    model_name = 'resnet18'
    split = 0.1

    files = get_img_files(path)
    num_train = int(len(files) * (1 - split))

    shuffle(files)

    # split data to train and eval
    train_files = files[:num_train]
    eval_files = files[num_train:]

    # define appropriate transforms:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # cerate dataset
    # train_set = BrainSet(image_files=train_files, root=path, transform=transform)
    # eval_set = BrainSet(image_files=eval_files, root=path, transform=transform)

    """
    # create dataloader
    train_dl = torch.utils.data.DataLoader(train_set,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=1)

    eval_dl = torch.utils.data.DataLoader(eval_set,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)

    """
    data_Set = ImageFolder(path, transform=transform)
    train_dl = torch.utils.data.DataLoader(data_Set,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=1)

    # run dummy data to encrypt model
    model = get_model(model_name, 2)
    model.train()

    iter_dl = iter(train_dl)

    for dummy_input, target in iter_dl:
        # encrypt model:

        encrypted_model = crypten.nn.from_pytorch(model, dummy_input=dummy_input)
        encrypted_model.encrypt()

        break

    encrypted_model.encrypt()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # loss
    criterion = nn.CrossEntropyLoss()

    losses = []
    for idx, sample in enumerate(train_dl):
        # preprocess sample:
        image, target = sample
        import pdb;
        pdb.set_trace()
        # image.require_grad = True
        image, target = Variable(image, requires_grad=True), Variable(target)

        import pdb;
        pdb.set_trace()
        # perform inference using encrypted model on encrypted sample:
        # encrypted_image = AutogradCrypTensor(image)
        encrypted_image = crypten.cryptensor(image)
        pdb.set_trace()

        # encrypted_image._tensor.requires_grad = True

        ####################
        # decrypted forward pass
        out = model(image)

        optimizer.zero_grad()
        loss = criterion(out, target)

        import pdb;
        pdb.set_trace()

        ##################
        # encrypted forward pass
        encrypted_output = encrypted_model(encrypted_image)
        print('got prediction')
        # measure accuracy of prediction
        output = encrypted_output.get_plain_text()

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu.item())

        print(idx, loss.cpu.item())


def example():
    skip_plaintext = False
    start_epoch = 0
    epochs = 3
    epochs = 1
    start_epoch = 0
    batch_size = 1
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-6
    print_freq = 10
    model_location = ""
    resume = False
    evaluate = True
    seed = None
    skip_plaintext = False
    context_manager = None
    best_prec1 = 0

    crypten.init()

    # setup
    path = '/Users/lukassanner/Downloads/brain-mri-images-for-brain-tumor-detection/'
    model_name = 'resnet18'
    split = 0.1

    model = get_model(model_name='resnet18', num_classes=2)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # loss
    criterion = nn.CrossEntropyLoss()

    files = get_img_files(path)
    num_train = int(len(files) * (1 - split))

    shuffle(files)

    # split data to train and eval
    train_files = files[:num_train]
    eval_files = files[num_train:]

    # define appropriate transforms:
    transform = transforms.Compose(
        [
            transforms.Resize(244),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_Set = ImageFolder(path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data_Set,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)

    data_Set = ImageFolder(path, transform=transform)
    val_loader = torch.utils.data.DataLoader(data_Set,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=1)

    # define loss function (criterion) and optimizer
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, print_freq)

        # remember best prec@1 and save checkpoint

        best_prec1 = max(prec1, best_prec1)
        is_best = prec1 > best_prec1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "LeNet",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    input_size = get_input_size(val_loader, batch_size)
    private_model = construct_private_model(input_size, model)

    validate_side_by_side(val_loader, plaintext_model=model, private_model=private_model)


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        output = model(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch}/{i} loss {loss}")


def validate_side_by_side(val_loader, plaintext_model, private_model):
    """Validate the plaintext and private models side-by-side on each example"""
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    softmax = nn.Softmax(dim=1)
    correct = 0

    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output for plaintext
            output_plaintext = plaintext_model(input)

            # encrypt input and compute output for private
            # assumes that private model is encrypted with src=0
            input_encr = encrypt_data_tensor_with_src(input)
            output_encr = private_model(input_encr)
            p = softmax(output_plaintext)
            p, predicted = p.data.max(1)

            correct += (predicted == target).sum().item()
            # log all info

            total += target.size(0)

            print(f"Example {i}\t target = {target}")
            print(f"Plaintext:{output_plaintext}")
            print(f"Encrypted:\n{output_encr.get_plain_text()}\n")
            print(f"predicted: {predicted}")
            print(f"confidence: {p}")
            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')

            # only use the first 1000 examples
            if i > 100:
                break


def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader))
    return input.size()


def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = get_model(model_name='resnet18', num_classes=2)
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def validate(val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                    input
            ):
                input = encrypt_data_tensor_with_src(input)
            # compute output
            output = model(input)
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            print(f"validate {i}, loss {loss.data}")
    return loss.data


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    # only save from rank 0 process to avoid race condition
    rank = comm.get().get_rank()
    if rank == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    example()
