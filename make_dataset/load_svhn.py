import numpy as np
import scipy.io

def load_svhn(dataset_dir='./SVHN/'):
    # Load training data
    train_data = scipy.io.loadmat(dataset_dir + 'train_32x32.mat')
    X_train = np.transpose(train_data['X'], (3, 0, 1, 2))  # Rearrange dimensions to N,H,W,C
    y_train = train_data['y'].flatten()
    y_train[y_train == 10] = 0  # Replace label 10 with 0 for consistency

    # Load test data
    test_data = scipy.io.loadmat(dataset_dir + 'test_32x32.mat')
    X_test = np.transpose(test_data['X'], (3, 0, 1, 2))  # Rearrange dimensions to N,H,W,C
    y_test = test_data['y'].flatten()
    y_test[y_test == 10] = 0  # Replace label 10 with 0 for consistency

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_svhn()
    print('==========================')
    print('Dataset information')
    print('train_img.shape, train_label.shape: ')
    print(train_img.shape, train_label.shape)
    print('test_img.shape, test_label.shape: ')
    print(test_img.shape, test_label.shape)
    print('train_img.min(), train_img.max(): ')
    print(train_img.min(), train_img.max())
    print('train_label.min(), train_label.max(): ')
    print(train_label.min(), train_label.max())
    print('example: ')
    print(train_img[:5], train_label[:5])
    print('==========================')
