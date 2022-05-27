import src.utils.interface_file_io as file_io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image


def convert_jpg(directory_path):
    file_list = []
    file_list += file_io.get_all_file_path(directory_path, 'jpg')
    file_list += file_io.get_all_file_path(directory_path, 'JPG')
    file_list += file_io.get_all_file_path(directory_path, 'png')
    file_list += file_io.get_all_file_path(directory_path, 'jpeg')
    file_list += file_io.get_all_file_path(directory_path, 'bmp')

    for file in tqdm(file_list, desc="convert images ... "):
        if 'jpg' in file:
            pass
        im = Image.open(file).convert('RGB')
        temp = file.split('.')[:-1]
        temp = '.'.join(temp)
        temp = "{}.jpg".format(temp)
        im.save(temp, 'jpeg')



def extract_label(dataset_name, filename):
    select = None
    if dataset_name == 'food-101':
        temp = filename.split('/')[5]
        select = temp
    elif dataset_name == 'fruits-360':
        temp = filename.split('/')[6]
        select = temp
    elif dataset_name == 'vegetables':
        temp = filename.split('/')[5]
        select = temp
    else:
        temp = filename.split('/')[4]
        select = temp
    return select


def label_extractor(dataset_name, dataset_path):
    file_list = file_io.get_all_file_path(dataset_path, file_extension='jpg')

    label = []
    for file in file_list:
        label.append(extract_label(dataset_name, file))
    label = list(set(label))

    file_io.make_list2txt(label, '../../dataset/{}-label.txt'.format(dataset_name))

def make_train_test_list(dataset_name, dataset_path, dataset_label):
    file_list = file_io.get_all_file_path(dataset_path, 'jpg')
    print(file_list[0])
    labels = file_io.read_txt2list(dataset_label)
    dataset = {tick: [] for tick in labels}

    for file in tqdm(file_list):
        file_label = extract_label(dataset_name, file)
        dataset[file_label].append(file)

    print(len(dataset))
    train_dataset = []
    test_dataset = []

    for key, value in dataset.items():
        if len(value) > 10:
            train, test = train_test_split(value, test_size=0.20, random_state=777)
            train_dataset += train
            test_dataset += test
    print(len(train_dataset))
    print(len(test_dataset))
    print(test_dataset[:10])

    file_io.make_list2txt(train_dataset, '../../dataset/{}-train.txt'.format(dataset_name))
    file_io.make_list2txt(test_dataset, '../../dataset/{}-test.txt'.format(dataset_name))



if __name__ == '__main__':
    convert_jpg('../../dataset/V3')
    # label_extractor('V3', '../../dataset/V3')
    # make_train_test_list('V3', '../../dataset/V3', '../../dataset/V3-label.txt')
    # convert_jpg('../../dataset/fruits-360_dataset')
    # label_extractor('vegetables', '../../dataset/fruits-360_dataset')
    # make_train_test_list('fruits-360', '../../dataset/fruits-360_dataset', '../../dataset/fruits-360-label.txt')
    # convert_jpg('../../dataset/vegetables')
    # label_extractor('vegetables', '../../dataset/vegetables')
    # make_train_test_list('vegetables', '../../dataset/vegetables', '../../dataset/vegetables-label.txt')