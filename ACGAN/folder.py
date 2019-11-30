import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


# hair_blonde hair_orange hair_brown hair_black hair_blue hair_white 
# eye_brown eye_blue eye_green eye_black 
# face_African face_Asian face_Caucasian 
# with_glasses without_glasses
# 15 (6+4+3+2)

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
                attr = [int(at) for at in line[1:]] 
                hair_attr = attr[:6]
                eye_attr = attr[6:10]
                face_attr = attr[10:13]
                glasses_attr = attr[13:]
                if is_image_file(img_file):
                    images.append((img_file, hair_attr, eye_attr, face_attr, glasses_attr))
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
            tuple: (image, target) where target is class_index of the target class.
        """
        path, attr1, attr2, attr3, attr4 = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        attr1 = torch.FloatTensor(attr1)
        attr2 = torch.FloatTensor(attr2)
        attr3 = torch.FloatTensor(attr3)
        attr4 = torch.FloatTensor(attr4)

        return img, attr1, attr2, attr3, attr4

    def __len__(self):
        return self.imgs_length
