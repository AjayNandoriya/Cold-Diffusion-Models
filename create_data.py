import torchvision
import os
import errno
import shutil
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


def main():
    CelebA_folder = '/fs/cml-datasets/CelebA-HQ/images-128/' # change this to folder which has CelebA data

    ############################################# MNIST ###############################################
    trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True)
    root = './root_mnist/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


    trainset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True)
    root = './root_mnist_test/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


    ############################################# Cifar10 ###############################################
    trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True)
    root = './root_cifar10/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


    trainset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True)
    root = './root_cifar10_test/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


    ############################################# CelebA ###############################################
    root_train = './root_celebA_128_train_new/'
    root_test = './root_celebA_128_test_new/'
    del_folder(root_train)
    create_folder(root_train)

    del_folder(root_test)
    create_folder(root_test)

    exts = ['jpg', 'jpeg', 'png']
    folder = CelebA_folder
    paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    for idx in range(len(paths)):
        img = Image.open(paths[idx])
        print(idx)
        if idx < 0.9*len(paths):
            img.save(root_train + str(idx) + '.png')
        else:
            img.save(root_test + str(idx) + '.png')


def create_sem_data():
    SEM_folder = './SEM_dataset/train'
    in_fname = './SEM_dataset/SRAM_22nm.jpg'
    img = cv2.imread(in_fname, 0)
    sH,sW = 128,128
    H,W = img.shape
    Nx = np.arange(0,W-sW+1, sW//2)
    Ny = np.arange(0,H-sH+1, sH//2)
    for iy in range(0, H-sH+1, sH//2):
        for ix in range(0, W-sW+1, sW//2):
            img_c = img[iy:(iy+sH),ix:(ix+sW)]
            out_img_fname = os.path.join(SEM_folder, f'img_{iy}_{ix}.png')
            cv2.imwrite(out_img_fname, img_c)
            pass
    pass

if __name__ == '__main__':
    # main()
    create_sem_data()
    pass