import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print(device)

root = './data/PH2Dataset'
seed = 42
size = (128, 128)

# images = []
# masks = []
# for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
#     if root.endswith('_Dermoscopic_Image'):
#         images.append(imread(os.path.join(root, files[0])))
#     if root.endswith('_lesion'):
#         masks.append(imread(os.path.join(root, files[0])))
#
# # Resize
# X = [resize(x, size, mode='constant', anti_aliasing=True, ) for x in images]
# Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in masks]
#
# X = np.array(X, np.float32)
# Y = np.array(Y, np.float32)

#  make pandas df with table data
table_path = root + '/PH2_dataset.xlsx'

with pd.ExcelFile(table_path) as xlsx:
    df = pd.read_excel(xlsx, sheet_name='Folha1', header=12)

# shortening of column names
names = dict((name, name.split('\n')[0].replace(' ', '_')) for name in df.columns)
df.rename(columns=names, inplace=True)

# set file name as index
# df.set_index('Image_Name', inplace=True)

# encode (NaN | 'X') values to (0 | 1)
for c in ('Common_Nevus', 'Atypical_Nevus', 'Melanoma',
          'White', 'Red', 'Light-Brown', 'Dark-Brown',
          'Blue-Gray', 'Black'):
    df[c] = df[c].apply(lambda x: 1 if x == 'X' else 0)

# apply Categorical dtype
derm_features = pd.CategoricalDtype(categories=('T', 'AT', 'A', 'P'))
for c in ('Pigment_Network', 'Dots/Globules', 'Streaks', 'Regression_Areas', 'Blue-Whitish_Veil'):
    df[c] = pd.Categorical(df[c], dtype=derm_features)


class PH2Dataset(data.Dataset):

    def __init__(self, root_path, df: pd.DataFrame, transforms=None):
        super().__init__()
        self.root_path = root_path
        self.df = df
        self.transforms = transforms

        # self.image_folder = os.path.join(self.root_path, "images")
        # self.mask_folder = os.path.join(self.root_path, "masks")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):

        # if index not in range(0, len(self.file_list)):

        if index >= self.__len__():
            return self.__getitem__(np.random.randint(0, self.__len__()))

        item = self.df.iloc[index]

        image_path = f"{self.root_path}/PH2 Dataset images/{item.Image_Name}/" \
                     f"{item.Image_Name}_Dermoscopic_Image/{item.Image_Name}.bmp"
        mask_path = f"{self.root_path}/PH2 Dataset images/{item.Image_Name}/" \
                    f"{item.Image_Name}_lesion/{item.Image_Name}_lesion.bmp"

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask, self.df.iloc[index]

    def show_samples(self, n_col=3, n_row=3, random=True):
        fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
        i = 0
        for row in range(n_row):
            for col in range(n_col):
                ax = axs[row, col]
                if random:  # random samples
                    image, mask, _ = self[np.random.randint(0, len(self))]
                else:  # first N samples
                    image, mask, _ = self[i]
                    i += 1

                show_image_and_mask(image, mask, ax)


def generate_datasets(test_ratio=0.25, valid_ratio=0.15):
    train_ids, test_ids = train_test_split(tuple(range(len(df.index))),
                                           test_size=test_ratio,
                                           shuffle=True, random_state=42)
    train_ids, valid_ids = train_test_split(train_ids,
                                            test_size=valid_ratio / (1 - test_ratio),
                                            shuffle=True, random_state=42)

    train_df = df.iloc[train_ids, :]
    valid_df = df.iloc[valid_ids, :]
    test_df = df.iloc[test_ids, :]

    train_dataset = PH2Dataset(root, train_df)
    valid_dataset = PH2Dataset(root, valid_df)
    test_dataset = PH2Dataset(root, test_df)

    return train_dataset, valid_dataset, test_dataset


def show_image_and_mask(image, mask, ax=None, mask_alpha=0.2):
    if ax is None:
        ax = plt.axes()
    ax.imshow(np.array(image, dtype=np.uint8))
    # ax.imshow(np.array(mask, dtype=np.uint8), cmap='', alpha=mask_alpha)
    ax.contour(mask, levels=1, colors='deepskyblue')
    ax.grid(False)
    ax.axis('off')


def compare_masks(image, gt_mask, pd_mask, ax=None):
    if ax is None:
        ax = plt.axes()
    ax.imshow(np.array(image, dtype=np.uint8))
    # ax.imshow(np.array(mask, dtype=np.uint8), cmap='', alpha=mask_alpha)
    ax.contour(gt_mask, levels=1, colors='deepskyblue')
    ax.contour(pd_mask, levels=1, colors='orangered')
    ax.grid(False)
    ax.axis('off')
