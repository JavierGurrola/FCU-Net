import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from skimage.util import view_as_windows
from skimage.color import rgb2gray


def load_image(path, patch_size):
    r"""
    Loads and pads image such that when the image
    is split into patches, complete patches are kept
    :param path: str
        Path of the image
    :param patch_size: int
        Size of the patch side
    :return:
        Padded image
    """
    image = np.asarray(Image.open(path))

    h_mod, w_mod = np.mod(image.shape[:2], patch_size)      # Extra pixels according to patch size
    if h_mod > 0:                                           # Difference of pixels according to patch size
        h_mod = patch_size - h_mod
    if w_mod > 0:
        w_mod = patch_size - w_mod

    if h_mod % 2 == 0:                                      # Pad the image symmetrically if it's possible
        pad_top = pad_down = h_mod // 2
    else:
        pad_top, pad_down = h_mod // 2, h_mod // 2 + 1

    if w_mod % 2 == 0:
        pad_left = pad_right = w_mod // 2
    else:
        pad_left, pad_right = w_mod // 2, w_mod // 2 + 1

    if image.ndim == 3:
        return np.pad(image, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), mode='edge')
    else:
        return np.pad(image, ((pad_top, pad_down), (pad_left, pad_right)), mode='edge')


def data_augmentation(image):
    r"""
    Apply flips and 90º degrees rotation as data augmentation to the image.
    :param image: numpy array
        Original image to apply data augmentation
    :return: list
        Transformed images as numpy arrays
    """

    to_transform = [image, np.rot90(image, axes=(0, 1))]
    augmented_images_list = [image, np.rot90(image, axes=(0, 1))]

    for t in to_transform:
        t_ud = t[::-1, ...]
        t_lr = t[:, ::-1, ...]
        t_ud_lr = t_ud[:, ::-1, ...]
        augmented_images_list.extend([t_ud, t_lr, t_ud_lr])

    return augmented_images_list


def create_patches(image, patch_size, step):
    r"""
    Convert image in patches
    :param image: numpy array
        Original image to transform in patches
    :param patch_size: tuple
        Patch size along each dimension
    :param step: tuple
        Distance between patches along each dimension
    :return: list
        Image patches as numpy arrays
    """

    ndim = image.ndim
    image = view_as_windows(image, patch_size, step)
    h, w = image.shape[:2]
    if ndim == 3:
        image = np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))
    else:
        image = np.reshape(image, (h * w, patch_size[0], patch_size[1]))

    return list(image)


class DataSampler(Sampler):
    r"""
    Dataset sampler to train the model in sub-epochs.
    Args:
        data_source (torch Dataset): training dataset.
        num_samples (int): number of samples per epoch (sub-epoch).
    """
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self.rand = np.random.RandomState(0)
        self.perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        """
        Generates new iterator with sample index.
        :return: iterator
            Index of the training samples of the current epoch.
        """
        n = len(self.data_source)
        if self._num_samples is not None:
            while len(self.perm) < self._num_samples:
                perm = self.rand.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)
            idx = self.perm[:self._num_samples]
            self.perm = self.perm[self._num_samples:]
        else:
            idx = self.rand.permutation(n).astype('int32').tolist()

        return iter(idx)

    def __len__(self):
        return self.num_samples


class VesselDataset(Dataset):
    r"""
    Retinal fondus image dataset.
    Args:
        files (list): Tuples in the form ('image path', 'label path', 'mask path') of the paths of every data sample.
        patch_size (int): Size of the patch side.
        step_size (dict): Step size of the patches according to each dataset.
        keep_masked_only (bool): Keep patches that contain masked information only.
        keep_vessel_only (bool): Keep patches that contain vessel label only.
        augment (bool): Apply data augmentation when loading the images.
        transform (torchvision transforms): List of transforms applied to the images.
        verbose (bool): Show loading data status information.
    """
    def __init__(self, files, patch_size, step_size, keep_masked_only=False, keep_vessel_only=False,
                 augment=False, transform=None, verbose=False):
        self.files = files
        self.patch_size = patch_size
        self.step_size = step_size
        self.keep_masked_only = keep_masked_only
        self.keep_vessel_only = keep_vessel_only
        self.augment = augment
        self.transform = transform
        self.verbose = verbose
        self.dataset = {'image': [], 'label': [], 'mask': []}
        self.dataset_counts = {'drive': 0, 'stare': 0, 'chase': 0}
        self.load_dataset()

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        image, label = self.dataset.get('image')[idx], self.dataset.get('label')[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample.get('image'), sample.get('label')

    def load_dataset(self):
        image_patch_size = (self.patch_size, self.patch_size, 3)
        mask_patch_size = (self.patch_size, self.patch_size)
        iterator = tqdm(self.files) if self.verbose else self.files

        for image_file, label_file, mask_file in iterator:
            image = load_image(image_file, self.patch_size)
            label = load_image(label_file, self.patch_size)
            mask = load_image(mask_file, self.patch_size)

            # Keep masks and labels as one channel images
            if label.ndim == 3:
                label = rgb2gray(label)
            if mask.ndim == 3:
                mask = rgb2gray(mask)

            # Convert from [0, 255] to [0, 1] and store as unsigned 8-bit int
            label, mask = label / label.max(), mask / mask.max()
            label, mask = label.astype('uint8'), mask.astype('uint8')

            if 'DRIVE' in image_file:
                key = 'drive'
            elif 'CHASE' in image_file:
                key = 'chase'
            else:
                key = 'stare'

            image_step_size = (self.step_size[key], self.step_size[key], 3)
            mask_step_size = (self.step_size[key], self.step_size[key])
            image_patches = create_patches(image, image_patch_size, image_step_size)
            label_patches = create_patches(label, mask_patch_size, mask_step_size)
            mask_patches = create_patches(mask, mask_patch_size, mask_step_size)

            for image_patch, label_patch, mask_patch in zip(image_patches, label_patches, mask_patches):
                if self.keep_masked_only and not np.any(mask_patch):
                    continue
                if self.keep_vessel_only and not np.any(label_patch):
                    continue

                if self.augment:
                    image_augmented, label_augmented = data_augmentation(image_patch), data_augmentation(label_patch)
                else:
                    image_augmented, label_augmented = [image_patch], [label_patch]

                for image_aug, label_aug in zip(image_augmented, label_augmented):
                    # Save in PIL image format to use ImageEnhance for data augmentation.
                    self.dataset['image'].append(Image.fromarray(image_aug.astype('uint8')))
                    self.dataset['label'].append(Image.fromarray(label_aug.astype('uint8')))
                    self.dataset_counts[key] += 1
