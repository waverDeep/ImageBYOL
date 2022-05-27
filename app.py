from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import PIL

import src.models.model as model
import src.utils.interface_file_io as file_io
import src.trainers.trainer as trainer
import src.trainers.tester as tester

app = Flask(__name__)

image_path = './upload'
config = './config/F10-serving-ImageBYOL-ResNet18-AdamP.json'
label_list = file_io.read_txt2list('./dataset/V3-label.txt')

with open(config, 'r') as configuration:
    config = json.load(configuration)

print(">> load model ...")
pretext_model = model.load_model(config=config, model_name=config["pretext_model_name"],
                                 checkpoint_path=config['pretext_checkpoint'])
downstream_model = model.load_model(config=config, model_name='Downstream', checkpoint_path=config['downstream_checkpoint'])
downstream_model.encoder = pretext_model.online_encoder

downstream_model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
print("model parameters: {}".format(downstream_model_params))
print("{}".format(downstream_model))
downstream_model = downstream_model.cuda()


def check_key(input_key):
    if input_key == '2022y05m22d':
        return True
    else:
        return False



def check_version(input_version):
    if input_version == "1":
        return True
    else:
        return False


def save_files(file_list):
    filename_list = []
    filepath_list = []
    for file in file_list:
        filename = secure_filename(file.filename)
        filename_list.append(filename)
        filename = os.path.join(image_path, filename)
        file.save(filename)
        filepath_list.append(filename)
    return filename_list, filepath_list

def inference(data):
    downstream_model.eval()
    # data = data.unsqueeze(0)
    if config['use_cuda']:
        data = data.cuda()
    with torch.no_grad():
        prediction = downstream_model(data)
        _, predicted = torch.max(prediction.data, 1)
    predicted = predicted.cpu().numpy()
    output = []
    for pick in predicted:
        output.append(label_list[pick])
    return output


def load_inference_data(file):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(640),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    data = PIL.Image.open(file)
    data = image_transforms(data)
    return data


def load_datapack(filelist):
    datapack = []
    image_transforms = transforms.Compose(
        [
            transforms.Resize(640),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    for file in filelist:
        data = PIL.Image.open(file)
        data = image_transforms(data)
        datapack.append(data)
    return torch.stack(datapack)




def main(file):
    data = load_inference_data(file)
    print(data.size())
    output = inference(data)
    print(output)
    return output


@app.route('/',)
def predict():
    return render_template('index.html')


@app.route('/multiple_predict', methods=['POST'])
def multiple_predict():
    response_part = {"status":"failure", "response_data":{"message": "check querystring parameters"}}
    if not check_key(request.values.get("appKey")):
        response_part['status'] = 'failure'
        response_part['response_data'] = {"message": "appKey error"}
        print(response_part)
        return response_part
    if not check_version(request.values.get("version")):
        response_part['status']='failure'
        response_part['response_data'] = {"message": "version error"}
        print(response_part)
        return response_part

    image_files = request.files.getlist("files")
    if len(image_files) == 0 or image_files[0].filename == '':
        response_part['status'] = 'failure'
        response_part['response_data'] = {"message": "image files are not exist"}
        print(response_part)
        return response_part
    filename_list, filepath_list = save_files(image_files)

    datapack = load_datapack(filepath_list)
    model_predicts = inference(datapack)
    response_predicts = {}
    # for i in range(len(filename_list)):
    #     model_predicts.append("output {}".format(i))

    for name, result in zip(filename_list, model_predicts):
        response_predicts[name]=result

    if len(response_predicts) is not 0:
        response_part['status'] = 'success'
        response_part['response_data'] = {"predicts": response_predicts}
        print(response_part)
        return response_part
    return response_part




@app.route('/')
def home():
    return 'Hello, World!'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6014, debug=True)