import os
import yaml
import torch
import numpy as np
from PIL import Image
from os.path import join
from model import FCUnet
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from skimage import io
from skimage.color import rgb2gray

from utils import build_ensemble, separate_ensemble, predict_ensemble, mod_pad, mod_crop, correct_model_dict
from metrics import get_sensitivity_score, get_specificity_score, get_masked_data


# def get_scores(prefix, )


def predict(model, image_dataset, label_dataset, mask_dataset, device, padding, results_path):
    metrics = {
        'F1': [], 'ACC': [], 'TPR': [], 'TNR': [], 'AUC': [],
        'ens F1': [], 'ens ACC': [], 'ens TPR': [], 'ens TNR': [], 'ens AUC': []
    }
    y_pred, y_pred_ens = [], []
    y_pred_scores, y_pred_ens_scores = [], []

    for i, (image, label, mask) in enumerate(zip(image_dataset, label_dataset, mask_dataset)):
        if padding:
            image, size = mod_pad(image, 8)
        else:
            image, label, mask = mod_crop(image, 8), mod_crop(label, 8), mod_crop(mask, 8)
            size = image.shape[:2]

        image = build_ensemble(image, normalize=True)

        with torch.no_grad():
            y_hat_ens = predict_ensemble(model, image, device)
            y_hat_ens, y_hat, y_hat_ens_score, y_hat_score = separate_ensemble(y_hat_ens, device, return_single=True)

            if padding:
                y_hat = y_hat[:size[0], :size[1], ...]
                y_hat_score = y_hat_score[:size[0], :size[1], ...]
                y_hat_ens = y_hat_ens[:size[0], :size[1], ...]
                y_hat_ens_score = y_hat_ens_score[:size[0], :size[1], ...]

            y_pred.append(y_hat)
            y_pred_ens.append(y_hat_ens)
            y_pred_scores.append(y_hat_score)
            y_pred_ens_scores.append(y_hat_ens_score)

            # Mask data and get metrics
            label_masked, y_hat_masked = get_masked_data(label, y_hat, mask)
            y_hat_score = get_masked_data(label, y_hat_score, mask)[1]
            y_hat_ens_masked = get_masked_data(label, y_hat_ens, mask)[1]
            y_hat_ens_score = get_masked_data(label, y_hat_ens_score, mask)[1]

            metrics['F1'].append(f1_score(label_masked, y_hat_masked))
            metrics['ACC'].append(accuracy_score(label_masked, y_hat_masked))
            metrics['AUC'].append(roc_auc_score(label_masked, y_hat_score))
            metrics['TPR'].append(get_sensitivity_score(label_masked, y_hat_masked))
            metrics['TNR'].append(get_specificity_score(label_masked, y_hat_masked))
            metrics['ens F1'].append(f1_score(label_masked, y_hat_ens_masked))
            metrics['ens ACC'].append(accuracy_score(label_masked, y_hat_ens_masked))
            metrics['ens AUC'].append(roc_auc_score(label_masked, y_hat_ens_score))
            metrics['ens TPR'].append(get_sensitivity_score(label_masked, y_hat_ens_masked))
            metrics['ens TNR'].append(get_specificity_score(label_masked, y_hat_ens_masked))

            message = 'Image:{} - TPR:{:.4f} - TNR:{:.4f} - AUC:{:.4f} - ACC:{:.4f} - F1:{:.4f} - '\
                      'ENS: - TPR:{:.4f} - TNR:{:.4f} - AUC:{:.4f} - ACC:{:.4f} - F1:{:.4f}'

            print(message.format(i + 1, metrics['TPR'][-1], metrics['TNR'][-1], metrics['AUC'][-1], metrics['ACC'][-1],
                                 metrics['F1'][-1], metrics['ens TPR'][-1], metrics['ens TNR'][-1],
                                 metrics['ens AUC'][-1], metrics['ens ACC'][-1], metrics['ens F1'][-1]))

    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        for i in range(len(y_pred)):
            y_hat, y_hat_ens = (255 * y_pred[i]).astype('uint8'), (255 * y_pred_ens[i]).astype('uint8')
            y_hat, y_hat_ens = np.squeeze(y_hat), np.squeeze(y_hat_ens)

            name_template = '{}_{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}{}.png'
            image_name = name_template.format(i + 1, metrics['TPR'][i], metrics['TNR'][i], metrics['AUC'][i],
                                              metrics['ACC'][i], metrics['F1'][i], '')
            io.imsave(os.path.join(results_path, image_name), y_hat)

            image_name = name_template.format(i + 1, metrics['ens TPR'][i], metrics['ens TNR'][i], metrics['ens AUC'][i],
                                              metrics['ens ACC'][i], metrics['ens F1'][i], '_ens')

            io.imsave(os.path.join(results_path, image_name), y_hat_ens)


    for metric, values in metrics.items():
        metrics[metric] = np.mean(values)

    return metrics


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    # Define and load pretrained model
    model_params = config['model']
    test_params = config['test']
    datasets = test_params['datasets']
    model_path = join(test_params['pretrained models path'], 'model_{}.pth')
    model = FCUnet(**model_params)

    if len(datasets) == 1:
        model_path = model_path.format(datasets[0])
    else:
        model_path = model_path.format('all')
    device = torch.device(test_params['device'])
    print("Using device: {}".format(device))

    state_dict = torch.load(model_path)
    state_dict = correct_model_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    for dataset in test_params['datasets']:
        print('Dataset: {}'.format(dataset))

        images_txt = join(test_params['testing files path'], dataset + '_test_images.txt')
        labels_txt = join(test_params['testing files path'], dataset + '_test_labels.txt')
        masks_txt = join(test_params['testing files path'], dataset + '_test_masks.txt')

        with open(images_txt, 'r') as f_images, open(labels_txt, 'r') as f_labels, open(masks_txt, 'r') as f_masks:
            images_files = list(map(lambda file: join(test_params['dataset path'], file), f_images.read().splitlines()))
            labels_files = list(map(lambda file: join(test_params['dataset path'], file), f_labels.read().splitlines()))
            masks_files = list(map(lambda file: join(test_params['dataset path'], file), f_masks.read().splitlines()))

        test_images, label_images, mask_images = [], [], []

        for i_file, l_file, m_file in zip(images_files, labels_files, masks_files):
            image = np.array(Image.open(i_file))
            label = np.array(Image.open(l_file))
            mask = np.array(Image.open(m_file))

            if label.ndim == 3:
                label = rgb2gray(label)
            if mask.ndim == 3:
                mask = rgb2gray(mask)

            label, mask = label / label.max(), mask / mask.max()
            label, mask = label.astype('uint8'), mask.astype('uint8')

            test_images.append(image)
            label_images.append(label)
            mask_images.append(mask)

        if test_params['save images']:
            save_path = join(test_params['results path'], dataset)
        else:
            save_path = None

        metrics = predict(model, test_images, label_images, mask_images, device, test_params['padding'], save_path)
        message = 'Simple: TPR:{TPR:.4f} - TNR:{TNR:.4f} - AUC:{AUC:.4f} - ACC:{ACC:.4f} - F1:{F1:.4f} ' \
                  'ENS: TPR:{ens TPR:.4f} - TNR:{ens TNR:.4f} - AUC:{ens AUC:.4f} - ACC:{ens ACC:.4f} - F1:{ens F1:.4f}'
        print(message.format(**metrics))
