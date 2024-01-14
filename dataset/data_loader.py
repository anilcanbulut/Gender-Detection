import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GenderDataset(Dataset):
    def __init__(self, root_dir, input_dim=(100, 100, 3), split="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.input_dim = input_dim
        self.split = split
        self.images = []
        self.labels = []

        for label, class_dir in enumerate(['male', 'female']):
            class_folder = os.path.join(root_dir, class_dir)
            for img_file in os.listdir(class_folder):
                self.images.append(os.path.join(class_folder, img_file))
                self.labels.append(label)
        
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        image = cv2.imread(img_path)

        is_gray = True if self.input_dim[2] == 1 else False

        image = cv2.resize(image, (self.input_dim[0], self.input_dim[1]))

        if is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image) 

        if self.split == "train":
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)

        label = self.labels[idx]
        return image, label