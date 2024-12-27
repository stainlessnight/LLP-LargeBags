import os
import numpy as np
from PIL import Image

def load_data(dataset_dir, categories):
    images = []
    labels = []

    for label, category in enumerate(categories):
        category_dir = os.path.join(dataset_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith('.tif'):
                img_path = os.path.join(category_dir, filename)
                img = Image.open(img_path)
                img = img.resize((224, 224))
                images.append(np.array(img))
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def load_nct_crc_data():
    train_dir = './NCT-CRC-HE-100K/'
    test_dir = './CRC-VAL-HE-7K/'
    categories = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    train_img, train_label = load_data(train_dir, categories)
    test_img, test_label = load_data(test_dir, categories)

    return train_img, train_label, test_img, test_label