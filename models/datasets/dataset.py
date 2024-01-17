
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from ..utils.conf import *

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)  # list of image file names
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # convert image to RGB
        if self.transform:
            image = self.transform(image)
        return image


# Image transformations
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize images to 512 x 512
    transforms.ToTensor(),  # convert PIL image to tensor
])

# Initialize dataset
dataset = ImageDataset(img_dir='/path/to/images', transform=transform)

# Initialize data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)