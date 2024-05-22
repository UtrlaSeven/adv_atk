import torchvision.transforms as transforms
from rich import print
import time
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn as nn
import torchattacks
import numpy as np
import argparse

from decimal import Decimal, getcontext
from rich.progress import Progress

#Sets global floating point precision


getcontext().prec = 20
epsilon = Decimal(8) / Decimal(255)
epsilon = epsilon.quantize(Decimal('1.000000000'))
def create_dataloader(dataset_name, batch_size, shuffle, if_train=True, if_norm=False):
    '''
    Create dataloader for pytorch models

    Args:
        dataset_name(str): name of dataset like 'CIFAR10'
        batch_size(int):
        shuffle(bool): if shuffle the dataloader
        if_train(bool): if use the training dataset
        if_norm(bool): if normalize the images

    Return:
        Dataloader
    '''
    time1 = time.time()
    if if_norm == True:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if dataset_name =='CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=if_train, download=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    time2 = time.time()
    print('[green]Dataloader for[bold blue] {}_train_{} [/bold blue]is ready, cost[bold blue] {}[/bold blue]s [/green]\n'.format(dataset_name, if_train, round(time2-time1,4)))
    return dataloader



def load_model(device,model_name,dataset_name,model_path):
    '''
    load the fine-tuning paramaters into a network model on device

    Args:
        device:
        model_name(str): name of the model, such as 'resnet50'
        dataset_name(str): name of dataset like 'CIFAR10'
        model_path(str): Absolute path to the model parameters file

    Return:
        model
    '''
    time1 = time.time()

    if model_name == 'resnet50' and dataset_name == "CIFAR10":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)

    time2 = time.time()
    print('[green]Loading model[bold blue] {} on {} [/bold blue]is ready,\
 cost[bold blue] {}[/bold blue]s [/green]\n'.format(model_name, device, round(time2-time1,4)))
    return model


def generate_attack_models(device,eps,attack_type=None, attack_model=None):
    '''
    load a attack model on device

    Args:
        device:
        eps(float):epsilon of attack model
        attack_type(str): name of the attack algorithm, such as 'PGD'
        attack_model(nn.Module): The model to attack

    Return:
        attack(nn.Module)
    '''
    if attack_type == "FGSM":
        attack = torchattacks.FGSM(model=attack_model, eps=eps)

    elif attack_type == "PGD":
        attack = torchattacks.PGD(model=attack_model,eps=eps)

    attack.set_device(device)
    print(f'[bold green]Successfully load {attack_type} to: [bold magenta]{attack.device}')
    return attack


def minkowski_distance(data, query_point, p=2):
    if p == float('inf'):
        distances = np.max(np.abs(data - query_point), axis=1)
    else:
        distances = np.sum(np.abs(data - query_point) ** p, axis=1) ** (1 / p)
    return distances

def precise_minkowski_distance(u, v, p):
    """
    Precisely calculate the Minkowski distance using the decimal library.

    Parameters:
        u(numpy.ndarray) : The first vector, a numpy ndarray.
        v(numpy.ndarray) : The second vector, a numpy ndarray.
        p(Decimal()) : The order of the Minkowski distance. If p is infinity, Chebyshev distance will be calculated.

    Returns:
    The precisely calculated Minkowski distance.
    """
    if p == Decimal('inf'):
        return max([Decimal(str(abs(ui - vi))) for ui, vi in zip(u, v)])

    distance = Decimal(0)
    for ui, vi in zip(u, v):
        diff = Decimal(ui) - Decimal(vi)
        distance += diff ** Decimal(p)
    
    # 计算p次根
    return distance ** (Decimal(1) / Decimal(p))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial attack')
    parser.add_argument('--attack_type', type=str, choices=['FGSM', 'PGD'],
                        required=True, help='atteck algorithm type')
    parser.add_argument('--model', type=str, choices=['resnet50'],
                        default='resnet50', help='atteck model')
    parser.add_argument('--dataset', type=str, choices=['CIFAT10'],
                        default='CIFAR10', help='dataset')
    args = parser.parse_args()

    batch_size = 64
    if_train = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    dataset_name = args.dataset
    attack_type = args.attack_type
    model_path=f'./models_pth/{dataset_name}/{model_name}_noNorm_0.8508.pth'


    origin_dataloader = create_dataloader(dataset_name=dataset_name,
                                         batch_size=batch_size, shuffle=False,
                                         if_train=if_train, if_norm=False)

    model = load_model(device,model_name,dataset_name,model_path)

    attack = generate_attack_models(device,attack_type=attack_type,
                                    eps=0.031372549, attack_model=model)

    targets_list = []
    original_labels_list = []
    adversarial_labels_list = []
    norm_infinity_list = []

    time1 = time.time()
    with Progress() as progress:
        task = progress.add_task(f"[red]generating result of {dataset_name}_{if_train}...",
                                 total=len(origin_dataloader)*batch_size)
        for i,(original_images, targets) in enumerate(origin_dataloader):

            original_images = original_images.to(device)
            adv_images = attack(original_images, targets)
            #print(original_images.shape,adv_images.shape,original_images.device, adv_images.device)

            original_output = model(original_images)
            _, original_predictions = torch.max(original_output,1)

            adversarial_output = model(adv_images)
            _, adversarial_predictions = torch.max(adversarial_output,1)

            targets_list.extend(targets.tolist())
            original_labels_list.extend(original_predictions.tolist())
            adversarial_labels_list.extend(adversarial_predictions.tolist())

            for ori_image,adv_image in zip(original_images,adv_images):
                ori_image, adv_image = ori_image.mean(dim=0), adv_image.mean(dim=0)
                ori_image, adv_image = ori_image.view(-1), adv_image.view(-1)
                ori_image, adv_image = ori_image.cpu().numpy(), adv_image.cpu().numpy()
                #print(ori_image.shape, adv_image.shape)
                norm_infinity = precise_minkowski_distance(ori_image,adv_image,p=Decimal('inf'))
                norm_infinity_list.append(str(norm_infinity))
            
                progress.update(task, advance=1)

    time2 = time.time()

    print('[green]Generate classification result by[bold blue] {}[/bold blue] on [bold blue]{} [/bold blue]is ready,\
 cost[bold blue] {}[/bold blue]s [/green]\n'.format(model_name, device, round(time2-time1,4)))
    print(f'[red]length of txt is {len(targets_list)} {len(adversarial_labels_list)} {len(original_labels_list)} {len(norm_infinity_list)}')

    with open(f'./predictions_{attack_type}_noNorm.txt', 'w') as f:
        for i, (label, ori_label, adv_label,inf_norms) in enumerate(zip(targets_list,original_labels_list,adversarial_labels_list,norm_infinity_list)):
            write_line = '{:06d} {} {} {} {}'.format(i,label,ori_label,adv_label,inf_norms)
            f.write(str(write_line)+'\n')