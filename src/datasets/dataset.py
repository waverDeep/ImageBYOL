from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as vision_dataset
import src.utils.interface_file_io as file_io
import torchvision.transforms as transforms
import torch.nn as nn
import PIL


def get_dataloader(config, mode='train', transform=None):
    dataset = None
    dataset_type = config['dataset_type']

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
    )
    return dataloader, dataset


class Food101Dataset(Dataset):
    def __init__(self, dataset_path):
        super(Food101Dataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class FoodCombiniationDataset(Dataset):
    def __init__(self, config, dataset_path, mode):
        super(FoodCombiniationDataset, self).__init__()
        self.file_list = file_io.read_txt2list(dataset_path)

        if mode == 'train':
            self.image_transforms = nn.Sequential(
                transforms.CenterCrop(config['crop_size'] + 256),
                transforms.RandomResizedCrop(config['crop_size']),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )
        else:
            self.image_transforms = nn.Sequential(
                transforms.CenterCrop(config['crop_size'] + 256),
                transforms.CenterCrop(config['crop_size']),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        data = PIL.Image.open(file[:4])
        image01 = self.image_transforms(data)
        image02 = self.image_transforms(data)
        return image01, image02
