import shutil

import crypten
import crypten.communicator as comm
import numpy as np
import torch
import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import ImageFolder

from model import get_model


# def get_model(model_name, num_classes):
#   return LeNet()

def train_epochs(args):
    epochs = args.max_epochs
    start_epoch = 0
    batch_size = args.batch_size
    lr = args.lr

    print_freq = 10

    best_prec1 = 0

    crypten.init()

    model = get_model(model_name=args.backbone, num_classes=args.num_classes)

    print(f"\n{'#'*40}")
    print(f"loaded {args.backbone} with {args.num_classes} classes")
    print(model)
    print(f"{'#'*40}\n")

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # loss
    criterion = nn.CrossEntropyLoss()

    if args.backbone == 'LeNet':
        input_dim = 32
    elif args.backbone == 'BigLeNet':
        input_dim = 64
    else:
        input_dim = 244
    # define appropriate transforms:
    transform = transforms.Compose(
        [
            transforms.Resize(input_dim),
            transforms.CenterCrop(input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = ImageFolder(args.source_dir_train, transform=transform)
    eval_set = ImageFolder(args.source_dir_eval, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=1)

    val_loader = torch.utils.data.DataLoader(eval_set,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=1)

    training_scores = []
    # define loss function (criterion) and optimizer
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        training_scores.append(prec1)

        # remember best prec@1 and save checkpoint

        best_prec1 = max(prec1, best_prec1)
        is_best = prec1 > best_prec1

        print(f"\n{'#'*40}")
        print(f"EPOCH {epoch} with score {np.mean(training_scores)}")
        print(f"{'#'*40}\n")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.backbone,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            is_best=is_best, filename=args.training_run_out
        )

    input_size = get_input_size(val_loader, batch_size)
    private_model = construct_private_model(input_size, model, args.backbone, args.num_classes)

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

        if i % print_freq == 0:
            print(f"train: {epoch}.{i} loss {loss}")


def validate_side_by_side(val_loader, plaintext_model, private_model):
    """Validate the plaintext and private models side-by-side on each example"""
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    softmax = nn.Softmax(dim=1)
    correct = 0
    scores = []
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

            score = accuracy(predicted, target)
            scores.append(score)

            print(f"Example {i}")
            print(f"target:\t\t {target}")
            print(f"predicted:\t {predicted}")
            print(f"confidence: {p}")

            print(f"Plaintext:\n{output_plaintext}")
            print(f"Encrypted:\n{output_encr.get_plain_text()}\n")
            print(f'Accuracy of the network on the 10000 test images: {np.mean(scores)}')

            # only use the first 1000 examples
            if i > 3:
                break


def accuracy(pred, target):
    correct = (pred == target).sum().item()
    correct /= target.size(0)
    return correct


def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader))
    return input.size()


def construct_private_model(input_size, model, model_name: str, num_classes: int):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = get_model(model_name=model_name, num_classes=num_classes)
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


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    scores = []

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                    input
            ):
                input = encrypt_data_tensor_with_src(input)
            # compute output
            output = model(input)
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()

            p = softmax(output)
            p, predicted = p.data.max(1)

            score = accuracy(predicted, target)
            scores.append(score)
            loss = criterion(output, target)

            print(f"validate {i}, loss {loss.data}, score {np.mean(scores)}")
    return np.mean(scores)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    # only save from rank 0 process to avoid race condition
    print(f"\n{'#'*40}")
    print(f"model saved to {filename}")
    print(f"{'#'*40}\n")

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CrypTen Predoc Training")
    parser.add_argument('--source-dir-train', type=str, help='source dir to folder training set')
    parser.add_argument('--source-dir-eval', type=str, help='source dir to folder training set')
    parser.add_argument('--training-run-out', type=str, help='save model at ..')

    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone, network architecture")
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes to train on')

    parser.add_argument('--validate-encrypted', type=bool, default=False, help='validate on encrypted model and data')
    parser.add_argument('--max-epochs', type=int, default=5, help='maxi epochs to train')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')

    args = parser.parse_args()
    train_epochs(args)
