import numpy as np

class ImageNormalize(object):
    def __init__(self, max=250.0, min=-200.0):
        self.max = max
        self.min = min
    
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image[image<self.min] = self.min
        image[image>self.max] = self.max
        # image = (image - np.mean(image)) / np.std(image)
        image = (image - self.min) / (self.max - self.min)
        samples = {}
        samples['image'] = image
        samples['labels'] = labels
        return samples


class RandomRotFlip(object):

    def __call__(self, sample):
        cropp_img, cropp_tumor = sample['image'], sample['labels']
        # # k = np.random.randint(1, 3)
        # # image = np.rot90(image, k)
        # # label = np.rot90(label, k)
        # axis = np.random.randint(1, 3)
        # image = np.flip(image, axis=axis).copy()
        # labels = np.flip(labels, axis=axis).copy()
        flip_num = np.random.randint(0,8)
        if flip_num == 1:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
        elif flip_num == 2:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        elif flip_num == 3:
            cropp_img = np.rot90(cropp_img, k=1, axes=(2, 1))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(2, 1))
        elif flip_num == 4:
            cropp_img = np.rot90(cropp_img, k=3, axes=(2, 1))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(2, 1))
        elif flip_num == 5:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=1, axes=(2, 1))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(2, 1))
        elif flip_num == 6:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=3, axes=(2, 1))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(2, 1))
        elif flip_num == 7:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)

        return {'image': cropp_img, 'labels': cropp_tumor}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['labels']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            ph = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0.0)
            label = np.pad(label, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0.0)

        (d, w, h) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        d1 = np.random.randint(0, d - self.output_size[0])
        w1 = np.random.randint(0, w - self.output_size[1])
        h1 = np.random.randint(0, h - self.output_size[2])

        label = label[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]
        return {'image': image, 'labels': label}

class RandCropByClass(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['labels']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            ph = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (d, w, h) = image.shape

        d1 = np.random.randint(0, d - self.output_size[0])
        w1 = np.random.randint(0, w - self.output_size[1])
        h1 = np.random.randint(0, h - self.output_size[2])

        label = label[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]
        return {'image': image, 'labels': label}

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['labels']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            ph = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (d, w, h) = image.shape

        d1 = int(round((d - self.output_size[0]) / 2.))
        w1 = int(round((w - self.output_size[1]) / 2.))
        h1 = int(round((h - self.output_size[2]) / 2.))

        label = label[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], w1:w1 + self.output_size[1], h1:h1 + self.output_size[2]]

        return {'image': image, 'labels': label}