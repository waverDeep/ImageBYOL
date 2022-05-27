import src.utils.interface_file_io as file_io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import multiprocessing
import src.utils.interface_multiprocessing as mi
import src.utils.interface_file_io as io

def convert(file_list):
    for file in tqdm(file_list, desc="convert images ... "):
        if 'jpg' in file:
            pass
        im = Image.open(file).convert('RGB')
        temp = file.split('.')[:-1]
        temp = '.'.join(temp)
        temp = "{}.jpg".format(temp)
        im.save(temp, 'jpeg')

def convert_jpg(directory_path):
    file_list = []
    file_list += file_io.get_all_file_path(directory_path, 'jpg')
    file_list += file_io.get_all_file_path(directory_path, 'JPG')
    file_list += file_io.get_all_file_path(directory_path, 'png')
    file_list += file_io.get_all_file_path(directory_path, 'jpeg')
    file_list += file_io.get_all_file_path(directory_path, 'bmp')

    divide_num = multiprocessing.cpu_count() - 1
    file_list = io.list_divider(divide_num, file_list)
    processes = mi.setup_multiproceesing(convert, data_list=file_list)
    mi.start_multiprocessing(processes)


def make_train_test_list(dataset_name, dataset_path):
    file_list = []

    file_list += file_io.get_all_file_path(dataset_path, 'jpg')

    train, test = train_test_split(file_list, test_size=0.20, random_state=777)

    file_io.make_list2txt(train, './dataset/{}-train.txt'.format(dataset_name))
    file_io.make_list2txt(test, './dataset/{}-test.txt'.format(dataset_name))



if __name__ == '__main__':
    convert_jpg('./dataset/food_fullset')
    make_train_test_list('food_fullset', './dataset/food_fullset')
