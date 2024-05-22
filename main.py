import argparse
import torch
from utils.attack_utils import create_dataloader,load_model,generate_attack_models
from decimal import Decimal, getcontext
from rich import print
import time
from rich.progress import Progress
from utils.tree import precise_minkowski_distance, create_tree,minkowski_distance
#Sets global floating point precision
getcontext().prec = 20
epsilon = Decimal(8) / Decimal(255)
epsilon = epsilon.quantize(Decimal('1.00000000'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial attack')
    parser.add_argument('--attack_type', type=str, choices=['FGSM', 'PGD'],
                        required=True, help='atteck algorithm type')
    parser.add_argument('--model', type=str, choices=['resnet50'],
                        default='resnet50', help='atteck model')
    parser.add_argument('--dataset', type=str, choices=['CIFAT10'],
                        default='CIFAR10', help='dataset')
    args = parser.parse_args()

    batch_size = 20
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
    ori_image_list = []
    unmean_image_list = []
    

    for i, (images,targets) in enumerate(origin_dataloader):
        #print(images[0])
        unimean_images = images.view(images.shape[0],-1)
        unmean_image_list.append(unimean_images)
        summed_images = images.mean(dim=1)
        flattened_images = summed_images.view(images.shape[0],-1)
        ori_image_list.append(flattened_images)
        targets_list.extend(targets)
        #break
    final_tensor = torch.cat(ori_image_list,dim=0)
    print(final_tensor.shape)
    unmean_tensor = torch.cat(unmean_image_list,dim=0)
    print(unmean_tensor.shape)
    kd_Tree = create_tree(features=final_tensor)

    corrent_count = 0
    time1 = time.time()
    with Progress() as progress:
        task = progress.add_task('[red]drawing...',total=len(final_tensor))
        for i, (images,targets) in enumerate(origin_dataloader):
            images = images.to(device)
            adv_images = attack(images, targets)
            for j,adv_image in enumerate(adv_images):
                current_index = j+i*batch_size
                unflattened_adv_image = adv_image.view(-1)
                unflattened_adv_image = unflattened_adv_image.cpu().numpy()

                query_image = adv_image.mean(dim=0)

                flattened_image = query_image.view(-1)
                flattened_image = flattened_image.cpu().numpy()

                distance, indice = kd_Tree.query(flattened_image, k=1,p=float('inf'))
                
                if current_index == indice:
                    corrent_count += 1
                
                # che_dis = minkowski_distance(flattened_image,final_tensor[current_index].cpu().numpy(),p=float('inf'))
                # pre_che_dis = precise_minkowski_distance(flattened_image,final_tensor[current_index].cpu().numpy(),p=Decimal('inf'))
                # test_dis = minkowski_distance(unflattened_adv_image,unmean_tensor[current_index].cpu().numpy(),p=float('inf'))
                # pre_test_dis = precise_minkowski_distance(unflattened_adv_image,unmean_tensor[current_index].cpu().numpy(),p=float('inf'))
                # print(0.031372549,test_dis,pre_test_dis,distance,round(distance,9),che_dis ,pre_che_dis)
                progress.update(task, advance=1)
            #break
    time2 = time.time()
    print(corrent_count,round(time2-time1,6))
    print('end')
