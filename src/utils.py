import random
import torch
import numpy as np
from collections import OrderedDict


def mod_crop(image, mod):
    r"""
    Crops image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to crop.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy array
        Copped image
    """
    size = image.shape[:2]
    size = size - np.mod(size, mod)
    image = image[:size[0], :size[1], ...]

    return image


def mod_pad(image, mod):
    r"""
    Pads image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to pad.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy  array, tuple
        Padded image, original image size.
    """
    size = image.shape[:2]
    h, w = np.mod(size, mod)
    h, w = mod - h, mod - w
    if h != mod or w != mod:
        if image.ndim == 3:
            image = np.pad(image, ((0, h), (0, w), (0, 0)), mode='edge')
        else:
            image = np.pad(image, ((0, h), (0, w)), mode='edge')

    return image, size


def set_seed(seed=1):
    r"""
    Sets all random seeds.
    :param seed: int
        Seed value.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_ensemble(image, normalize=True):
    r"""
    Create image ensemble to estimate denoised image.
    :param image: numpy array
        Noisy image.
    :param normalize: bool
        Normalize image to range [0., 1.].
    :return: list
        Ensemble of noisy image transformed.
    """
    img_rot = np.rot90(image)
    ensemble_list = [
        image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image)),
        img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))
    ]

    ensemble_transformed = []
    for img in ensemble_list:
        if img.ndim == 2:  # Expand dims for channel dimension in gray scale.
            img = np.expand_dims(img.copy(), 0)  # Use copy to avoid problems with reverse indexing.
        else:
            img = np.transpose(img.copy(), (2, 0, 1))  # Channels-first transposition.
        if normalize:
            img = img / 255.

        img_t = torch.from_numpy(np.expand_dims(img, 0)).float()  # Expand dims again to create batch dimension.
        ensemble_transformed.append(img_t)

    return ensemble_transformed


def separate_ensemble(ensemble, device, return_single=False):
    r"""
    Apply inverse transforms to predicted image ensemble and average them.
    :param ensemble: list
        Predicted images, ensemble[0] is the original image,
        and ensemble[i] is a transformed version of ensemble[i].
    :param device: torch device

    :param return_single: bool
        Return also ensemble[0] to evaluate single prediction
    :return: tuple of numpy arrays
        (ensemble thresholded, ensemble score) or
        (ensemble thresholded, single prediction thresholded, ensemble score, single prediction score)
    """
    ensemble_np = []

    for img in ensemble:
        img = img.squeeze()  # Remove additional dimensions.
        if img.ndim == 3:  # Transpose if necessary.
            img = np.transpose(img, (1, 2, 0))

        ensemble_np.append(img)

    # Apply inverse transforms to vertical and horizontal flips.
    img = ensemble_np[0] + np.fliplr(ensemble_np[1]) + np.flipud(ensemble_np[2]) + np.fliplr(np.flipud(ensemble_np[3]))

    # Apply inverse transforms to 90º rotation, vertical and horizontal flips
    img = img + np.rot90(ensemble_np[4], k=3) + np.rot90(np.fliplr(ensemble_np[5]), k=3)
    img = img + np.rot90(np.flipud(ensemble_np[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble_np[7])), k=3)

    sigmoid = lambda z: 1. / (1. + np.exp(-z))

    img = sigmoid(img / 8.)  # Prediction over ensemble average
    img_pred = np.round(img)

    if return_single:  # Prediction over single image too
        img_single = sigmoid(ensemble_np[0])
        img_single_pred = np.round(img_single)
        return img_pred.astype('uint8'), img_single_pred.astype('uint8'), img, img_single
    else:
        return img_pred.astype('uint8'), img


def predict_ensemble(model, ensemble, device):
    r"""
    Predict batch of images from an ensemble.
    :param model: torch Module
        Trained CNN model.
    :param ensemble: list
        Images to estimate.
    :param device: torch device
        Device of the CNN model.
    :return: list
        Estimated images as numpy ndarray.
    """
    y_hat_ensemble = []

    for i, x in enumerate(ensemble):
        x = x.to(device)
        with torch.no_grad():
            y_hat = model(x)
            # y_hat_ensemble.append(y_hat.cpu().detach().numpy().astype('float32'))
            y_hat_ensemble.append(y_hat.cpu().detach().numpy().astype('float32'))

    return y_hat_ensemble


def correct_model_dict(state_dict):
    r"""
    Re-name module names in state_dict to remove 'module.' word.
    :param state_dict: OrderedDict
        model modules
    :return: OrderedDict
        renamed model modules
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict
