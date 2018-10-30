import os
import random
import csv
import random
import pandas as pd
import numpy as np
import scipy.misc
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['CellHistology']

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

class ToOneHot(object):

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, labels, n_class):

        labels = labels.type(torch.LongTensor)
        _, h, w = labels.size()
        onehot = torch.zeros(n_class, h, w)
        target = onehot.scatter_( self.dim, labels.data, torch.ones(n_class, h, w) )

        return target

class FromOneHot(object):

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, onehot):

        onehot = onehot.type(torch.FloatTensor)
        c, h, w = onehot.shape
        target = onehot.argmax(dim=self.dim).reshape(1, h, w).float()

        return target

class CellHistology(Dataset):

    def __init__(self, mode, root_dir):

        self.root_dir = root_dir
        self.mode = mode.lower()

        if mode.lower() == 'train':
            self.csv = os.path.join(root_dir, 'train.csv')
            self.slides = range(1,10)
        elif mode.lower() == 'val':
            self.csv = os.path.join(root_dir, 'val.csv')
            self.slides = range(10,12)
        else:
            raise ValueError

        self.mean_file = os.path.join(root_dir, 'mean.npy')
        self.mean = np.zeros((1,3))
        self.std_file = os.path.join(root_dir, 'std.npy')
        self.std = np.zeros((1,3))

        self._preprocess()

        self.data = pd.read_csv(self.csv)
        self.mean = self.mean/255.          # Convert from 0-255, to 0-1
        self.std = self.std/255.
        self.n_class = 2

        self.toPILImage = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(tuple(self.mean.T[0]),
                                              tuple(self.std.T[0]))
        self.normalizeInverse = NormalizeInverse(tuple(self.mean.T[0]),
                                                 tuple(self.std.T[0]))
        self.toOneHot = ToOneHot()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''

        '''

        image_name  = self.data.iloc[idx, 0]
        image       = scipy.misc.imread(image_name, mode='RGB')
        image       = Image.fromarray(image, 'RGB')

        label_name  = self.data.iloc[idx, 1]
        label       = np.load(label_name)
        label       *= 255
        label       = Image.fromarray(label, 'L')

        # only flip if training
        rhval = random.random()
        rvval = random.random()
        if rhval >= .5 and self.mode=='train':
            hflip = transforms.RandomHorizontalFlip(p=1.0)
        else:
            hflip = transforms.RandomHorizontalFlip(p=0.0)
        if rvval >= .5 and self.mode=='train':
            vflip = transforms.RandomVerticalFlip(p=1.0)
        else:
            vflip = transforms.RandomVerticalFlip(p=0.0)

        image = hflip(image)                     # apply hflip operation
        image = vflip(image)                     # apply hflip operation
        image_t = self.toTensor(image)          # convert to tensor
        image_t = self.normalize(image_t)       # mean center, and normalize

        label = hflip(label)                     # apply hflip operation
        label = vflip(label)                     # apply hflip operation
        label_t = self.toTensor(label)          # convert to tensor
        label_t = self.toOneHot(label_t, 2)
        # NOTE: no need to one hot because 2 classes and using BCELogits

        data = {'X': image_t, 'Y': label_t}

        return data

    def _preprocess(self):
        '''
            Create train.csv and val.csv partitions of the data set. Create
            mean.npy which contains a 3x1 vector of mean pixel values of the
            data set. ...
        '''

        with open(self.csv, 'w') as f:

            f.write("image,label\n")

            for slide in self.slides:
                num = "%02d"%(slide,)
                image_dir = os.path.join( self.root_dir, "Slide_" + num )
                label_dir = os.path.join( self.root_dir, "GT_" + num )

                for file in os.listdir(image_dir):

                    image_file = os.path.join(image_dir, file)
                    label_file = os.path.join(label_dir, file)

                    f.write("%s,%s.npy\n"%(image_file, label_file))

                    if os.path.exists(label_file + '.npy'):
                        continue

                    print("Parsing " + file + ".npy")

                    label = scipy.misc.imread(label_file)
                    label = label/255

                    np.save(label_file+'.npy', label)

                    print("Finished " + file + ".npy")


        # Calculate mean data set values
        if self.mode == 'val':
            if not os.path.exists(self.mean_file):
                raise RuntimeError("mean.npy not found. Create train set first to create mean.npy file.")
            else:
                self.mean = np.load(self.mean_file)
            if not os.path.exists(self.std_file):
                raise RuntimeError("std.npy not found. Create train set first to create std.npy file.")
            else:
                self.std = np.load(self.std_file)
        else:
            if os.path.exists(self.mean_file):
                print("Mean.npy already exists, skipping")
                self.mean = np.load(self.mean_file)
            else:
                print("Calculating mean...")

                mean = np.zeros((1,3))

                with open(self.csv, 'r') as f:
                    csvreader = csv.reader(f)
                    csvreader.next() # skip first line
                    cnt = 0
                    for row in csvreader:
                        image = scipy.misc.imread(row[0], mode='RGB')
                        mean += image.mean(axis=(0,1))
                        cnt +=1

                mean /= cnt
                mean = mean.T
                print("RGB Mean Pixel Values: ")
                print(mean)

                np.save(self.mean_file, mean)
                self.mean = mean

            if os.path.exists(self.std_file):
                print("std.npy already exists, skipping")
                self.std = np.load(self.std_file)
            else:
                print("Calculating std dev...")

                std = np.zeros((1,3))

                with open(self.csv, 'r') as f:
                    csvreader = csv.reader(f)
                    csvreader.next() # skip first line
                    cnt = 0
                    for row in csvreader:
                        image = scipy.misc.imread(row[0], mode='RGB')
                        std += np.square((image.mean(axis=(0,1)) - self.mean.T))
                        cnt +=1

                std /= cnt
                std = np.sqrt(std)
                std = std.T
                print("RGB Std Dev Pixel Values: ")
                print(std)

                self.std = std
                np.save(self.std_file, std)


    def visualize(self, image, label, prediction=None):

        fromOneHot = FromOneHot()
        label = fromOneHot(label)

        image = self.toPILImage(self.normalizeInverse(image))
        label = self.toPILImage(label)

        if prediction is not None:
            prediction = self.toPILImage(prediction)
            label = Image.composite( Image.new('RGB', label.size,
                                        color='rgb(0,255,0)'),
                                    Image.new('RGB', label.size),
                                    label
                                    )
            prediction = prediction.convert('RGB')
            composite = Image.blend(prediction, label, .15)

            merge = Image.new('RGB', (label.size[0]+image.size[0],
                             max((label.size[1], image.size[1]))) )
            merge.paste(image, (0,0))
            merge.paste(composite, (image.size[0], 0))
            merge.show()

            return image, label, prediction
        else:
            image.show()
            label.show()

            return image, label

# Test Code
if __name__ == "__main__":
    CTrain = CellHistology('train', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')
    CVal = CellHistology('val', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')

    data = CTrain[0]
    image = data['X']
    label = data['Y']
    #label *= 255
    #image = Image.fromarray(image, 'RGB')
    #label = Image.fromarray(label, 'L')
    CTrain.visualize(image, label)
