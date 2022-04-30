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
config = './config/F10-serving-ImageBYOL-AdamP.json'
label_list = file_io.read_txt2list('./dataset/INGD_V1-category.txt')

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


def inference(data):
    downstream_model.eval()
    data = data.unsqueeze(0)
    if config['use_cuda']:
        data = data.cuda()
    with torch.no_grad():
        prediction = downstream_model(data)
        _, predicted = torch.max(prediction.data, 1)
    predicted = predicted.cpu().numpy()
    output = label_list[predicted[0]]
    print(output)


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


def main(config, file):
    data = load_inference_data(file)
    print(data.size())
    inference(data)


@app.route('/',)
def predict():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def file_upload():
    gt_file = request.files.get('gt_file')

    gt_filename = secure_filename(gt_file.filename)
    gt_file.save(os.path.join(image_path, gt_filename))

    main(config, os.path.join(image_path, gt_filename))

    return render_template('predict.html')



@app.route('/')
def home():
    return 'Hello, World!'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6014, debug=True)