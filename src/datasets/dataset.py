from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as vision_dataset
import src.utils.interface_file_io as file_io
import torchvision.transforms as transforms
import torch.nn as nn
import PIL


def get_dataloader(config, mode='train', transform=None):
    dataset = None
    dataset_type = config['dataset_name']

    if dataset_type == 'FoodCombinationDataset':
        dataset = FoodCombinationDataset(config=config, dataset_path=config['{}_dataset'.format(mode)], mode=mode)
    elif dataset_type == 'INGD_V1':
        dataset = INGDDataset(config=config, directory_path=config['{}_dataset'.format(mode)], mode=mode, crop_size=512)
    elif dataset_type == 'INGD_V2':
        dataset = INGDDataset(config=config, directory_path=config['{}_dataset'.format(mode)], mode=mode,
                              crop_size=512)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=config['dataset_shuffle'],
        num_workers=config['num_workers'],
    )
    return dataloader, dataset


class FoodCombinationDataset(Dataset):
    def __init__(self, config, dataset_path, mode):
        super(FoodCombinationDataset, self).__init__()
        self.file_list = file_io.read_txt2list(dataset_path)

        if mode == 'train':
            self.image_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(config['crop_size'] + 256),
                    transforms.RandomResizedCrop(config['crop_size']),
                    transforms.ColorJitter(),
                    transforms.RandomAdjustSharpness(2),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=(0, 360)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(config['crop_size'] + 256),
                    transforms.CenterCrop(config['crop_size']),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        file = file[4:]
        data = PIL.Image.open(file)
        image01 = self.image_transforms(data)
        image02 = self.image_transforms(data)
        return image01, image02


class INGDDataset(Dataset):
    def __init__(self, config, directory_path, mode='train', crop_size=512):
        super(INGDDataset, self).__init__()
        self.label_list = file_io.read_txt2list('./dataset/INGD_V2.txt')
        self.file_list = file_io.read_txt2list(directory_path)
        if mode == 'train':
            self.image_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(config['crop_size'] + 256),
                    transforms.RandomResizedCrop(config['crop_size']),
                    transforms.ColorJitter(),
                    transforms.RandomAdjustSharpness(2),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=(0, 360)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(640),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # './dataset/INGD_V1/corn/194.jpg'
        file = self.file_list[x]
        label = file.split('/')[3]
        label = label.lower()
        label = label.replace(' ', "_")
        label = self.label_list.index(label)
        # have issue - png 4channel cannot load only supported 3 dimension
        data = PIL.Image.open(file)
        # data = data.type('torch.FloatTensor')
        data = self.image_transforms(data)
        return data, label