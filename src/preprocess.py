import os
import shutil
import cv2 as cv


def preprocess(src_root_path, target_root_path):
    r"""
    Preprocess the dataset (image, label and FOV mask) before training and test the model.
    :param src_root_path: str
        Root path of the raw dataset.
    :param target_root_path:
        Root path of the preprocessed dataset.
    :return: None
    """
    folders = ['images', '1st_manual', 'mask']

    for folder in folders:
        src_path = os.path.join(src_root_path, folder)
        target_path = os.path.join(target_root_path, folder)

        if folder == 'images':
            os.makedirs(target_path, exist_ok=True)
            files = os.listdir(src_path)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))    # Pre-processing method

            for file in files:
                image = cv.imread(os.path.join(src_path, file))             # File in format BGR
                g_channel = image[..., 1]
                g_clahe = clahe.apply(g_channel)                            # Apply preprocessing in the green channel only
                image[..., 1] = g_clahe
                cv.imwrite(os.path.join(target_path, file), image)
        else:
            shutil.copytree(src_path, target_path)                          # Direct copy label and mask


if __name__ == '__main__':
    datasets = ['DRIVE/training', 'DRIVE/test', 'STARE', 'CHASE']
    path_org = os.path.join('..', 'dataset')
    path_dest = os.path.join('..', 'dataset_processed')

    for dataset in datasets:
        dataset_path_org = os.path.join(path_org, dataset)
        dataset_path_dest = os.path.join(path_dest, dataset)
        os.makedirs(dataset_path_dest, exist_ok=True)
        preprocess(dataset_path_org, dataset_path_dest)
