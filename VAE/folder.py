import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(attr_file, image_dir):
    images = []
    image_length = 0
    with open(attr_file, 'r') as af:
        lines = af.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                image_length = int(line)
            elif idx > 1:
                line = line.split()
                img_file = os.path.join(image_dir, line[0])
                if is_image_file(img_file):
                    images.append(img_file)
                else:
                    print('{} not exist!'.format(img_file))
                    exit()
    return images, image_length


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None,loader=default_loader):
        self.attr_file = os.path.join(root, 'cartoon_attr.txt')
        self.image_dir = os.path.join(root, 'images/')
        imgs, imgs_length = make_dataset(self.attr_file, self.image_dir)

        self.imgs = imgs
        self.imgs_length = imgs_length

        self.root = root
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image : Image
        """
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.imgs_length
