import yaml
import torch
import numpy as np
from os.path import join
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info

from model import FCUnet
from data_management import VesselDataset, DataSampler
from train import fit_model
from transforms import ColorJitter, Rotate, ToTensor
from utils import set_seed


def main():
    # Load YAML configuration file with predefined parameters.
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    model_params, train_params, val_params = config['model'], config['train'], config['val']

    # Defining model and print the summary
    set_seed(0)
    model = FCUnet(**model_params)
    device = torch.device(train_params['device'])
    print("Using device: {}".format(device))

    if torch.cuda.device_count() > 1 and 'cuda' in device.type and train_params['multi gpu']:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')
    model = model.to(device)

    print('Model summary:')
    test_shape = (3, train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    checkpoints_path = train_params['checkpoint path']
    images_files, labels_files, masks_files = [], [], []
    for dataset in train_params['datasets']:
        checkpoints_path = checkpoints_path + '_' + dataset
        images_txt = join(train_params['training files path'], dataset + '_training_images.txt')
        labels_txt = join(train_params['training files path'], dataset + '_training_labels.txt')
        masks_txt = join(train_params['training files path'], dataset + '_training_masks.txt')

        with open(images_txt, 'r') as f_images, open(labels_txt, 'r') as f_labels, open(masks_txt, 'r') as f_masks:
            images_files += list(map(lambda file: join(train_params['dataset path'], file), f_images.read().splitlines()))
            labels_files += list(map(lambda file: join(train_params['dataset path'], file), f_labels.read().splitlines()))
            masks_files += list(map(lambda file: join(train_params['dataset path'], file), f_masks.read().splitlines()))

    # Validation dataset definition, one image per possible dataset.
    drive_val, chase_val, stare_val = False, False, False
    val_index = []
    index = np.random.permutation(len(images_files))
    for i in index:
        if 'DRIVE' in images_files[i] and not drive_val:
            val_index.append(i)
            drive_val = True
        elif 'CHASE' in images_files[i] and not chase_val:
            val_index.append(i)
            chase_val = True
        elif 'STARE' in images_files[i] and not stare_val:
            val_index.append(i)
            stare_val = True
        if drive_val and chase_val and stare_val:
            break

    train_files, val_files = [], []
    print('\nValidation images:')
    for i, (image, seg, mask) in enumerate(zip(images_files, labels_files, masks_files)):
        if i not in val_index:
            train_files.append((image, seg, mask))
        else:
            print(image)
            val_files.append((image, seg, mask))

    training_transform = transforms.Compose([
        ColorJitter(),
        Rotate(),
        ToTensor()
    ])

    print('\nLoading training dataset:')
    train_step_size = {
        'drive': train_params['patch size'] // 3,
        'stare': train_params['patch size'] // 2,
        'chase': train_params['patch size'] // 2
    }
    training_dataset = VesselDataset(train_files, train_params['patch size'], train_step_size, True, False,
                                     train_params['augment dataset'], training_transform, train_params['verbose'])

    print('\nLoading validation dataset:')
    val_step_size = {
        'drive': train_params['patch size'],
        'stare': train_params['patch size'],
        'chase': train_params['patch size']
    }
    validation_dataset = VesselDataset(val_files, val_params['patch size'], val_step_size, True, False, False,
                                       ToTensor(), train_params['verbose'])
    # Training in sub-epochs:
    print('Training patches:{}\nValidation patches:{}'.format(len(training_dataset), len(validation_dataset)))
    print(training_dataset.dataset_counts)
    n_samples = len(training_dataset) // train_params['dataset splits']
    n_epochs = train_params['epochs'] * train_params['dataset splits']
    sampler = DataSampler(training_dataset, num_samples=n_samples)

    data_loaders = {
        'train': DataLoader(training_dataset, train_params['batch size'], num_workers=train_params['workers'],
                            sampler=sampler),
        'val': DataLoader(validation_dataset, val_params['batch size'], num_workers=val_params['workers']),
    }

    # Optimization:
    # criterion = nn.BCELoss() # Requieres sigmoid activation
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = train_params['learning rate']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=train_params['weight decay'])
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=train_params['weight decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_params['scheduler step'],
                                             gamma=train_params['scheduler gamma'])

    # Train the model
    fit_model(model, data_loaders, criterion, optimizer, lr_scheduler, device, n_epochs, val_params['frequency'],
              checkpoints_path, 'model', train_params['verbose'])


if __name__ == '__main__':
    main()
