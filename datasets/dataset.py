import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from ..utils.conf import *

class JigsawDataset(Dataset):
    def __init__(self, csv_file, transform=None, test=False):
        """
        csv_file (string): csv file with all the images and labels.
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_frame = csv_file
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_PATH, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)
        labels = torch.tensor(self.data_frame.iloc[idx, 2:].astype(int).tolist(), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            labels = self.data_frame.iloc[idx, 2:].astype(int).tolist()
            return (image, labels)
        

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize images to 512 x 512
    transforms.ToTensor(),  # convert PIL image to tensor
])

# Split validation
train_df, valid_df = train_test_split(pd.read_csv(TRAIN_CSV), test_size=0.2, random_state=42) 

# Initialize dataset
TRAIN_DATASET = JigsawDataset(csv_file=train_df, transform=transform)
VALID_DATASET = JigsawDataset(csv_file=valid_df, transform=transform)
TEST_DATASET = JigsawDataset(csv_file=pd.read_csv(TEST_CSV), transform=transform, test=True)

# Initialize data loader
TRAIN_DATA_LOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
VALID_DATA_LOADER = DataLoader(VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False)
TEST_DATA_LOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False)