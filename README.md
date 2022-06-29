# Factorized U-net for Retinal Vessel Segmentation

This repository is for the RDUNet model proposed in the following conference paper:

[Javier Gurrola-Ramos](https://scholar.google.com.mx/citations?user=NuhdwkgAAAAJ&hl=es), [Oscar Dalmau](https://scholar.google.com.mx/citations?user=5oUOG4cAAAAJ&hl=es&oi=sra) and [Teresa E. Alarcón](https://scholar.google.com.mx/citations?user=gSUClZYAAAAJ&hl=es&authuser=1), ["Factorized U-net for Retinal Vessel Segmentation"](https://link.springer.com/chapter/10.1007/978-3-031-07750-0_17), Pattern Recognition. MCPR 2022. Lecture Notes in Computer Science, vol 13264. Springer, Cham. doi: [10.1007/978-3-031-07750-0_17](https://doi.org/10.1007/978-3-031-07750-0_17).

## Citation
If you use this paper work in your research or work, please cite our paper:

```
@inproceedings{gurrola2022factorized,
  title={Factorized U-net for Retinal Vessel Segmentation},
  author={Gurrola-Ramos, Javier and Dalmau, Oscar and Alarc{\'o}n, Teresa},
  booktitle={Mexican Conference on Pattern Recognition},
  pages={181--190},
  year={2022},
  organization={Springer}
}

```
![FCU-Net](https://github.com/JavierGurrola/FCU-Net/blob/main/figs/model.png)

## Dependencies
- Python 3.6
- PyTorch 1.8.0
- Torchvision 0.9.0
- Numpy 1.19.2
- Pillow 8.1.2
- ptflops 0.6.4
- tqdm 4.50.2
- scikit-learn 0.23.2
- scikit-image 0.17.2
- cv2 4.5.1
- yaml 5.3.1

## Pre-trained model and datasets
Pre-trained models are placed in this [folder](https://github.com/JavierGurrola/FCU-Net/blob/main/pretrained).

## Training

First, apply the preprocessing to the datasets by running the following command:

```python preprocess.py```

Default parameters used in the paper are set in the ```config.yaml``` file:

```
base filters: 16
dropout rate: 0.3
patch size: 48
batch size: 64
learning rate: 1.e-3
weight decay: 2.e-2
scheduler gamma: 0.975
scheduler step size: 10
epochs: 60
```

By modifying the ```config.yaml``` file, you can select the datasets you want to work with:

```
datasets:
  - 'drive'
  - 'stare'
  - 'chase'
```

To train the model use the following command:

```python main_train.py```

## Test

To test the model use the following command:

```python main_test.py```


## Contact

If you have any question about the code or paper, please contact francisco.gurrola@cimat.mx .
