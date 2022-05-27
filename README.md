# BYOL for Food Recognition

BYOL 기반 음식 사진 인식 모델 구축


## Description

다양한 환경과 다양한 형상의 음식, 식재료 사진들을 인식하기 위해서 self-supervised learning을 도입하였다. 
Computational limit이 있었기 때문에 negative sample이 필요하지 않은 BYOL architecture를 base로 하였다. 
Pretext task에서는 vegetables, foods, fruits를 포함하고 있는 open dataset을 모두 합쳐서 사용하였다.
Downstream task는 직접 수집한 데이터를 사용하였다.
이 모델은 Reciptopia application에 적용되며 사용자의 동의를 거친 후에 사용자가 직접 찍을 사진을 기반으로 모델을 최적화시켜나갈 계획이다.

## Getting Started

1. Make up own your configuration file.  (There is an pretext example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...

```
3. And then, start pretext task training!
```
python train.py
```


### Downstream task

Currently, only transfer learning is implemented in this project.
1. Make up own your configuration file.  (There is an transfer learning example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...
```

3. And then, start pretext task training!
4. 
```
python train.py
```


### Dependencies

* Linux Ubuntu, Nvidia Docker, Python
* torch
* torchvision
* adamp 0.3.0
* scikit-learn 1.0.2
* numpy 1.21.6
* tensorboard 2.8.0
* torch 1.12.0
* torchvision 0.13.0
* natsort 8.1.0


## Authors
[waverDeep](https://github.com/waverDeep)
and [FirstianB101](https://github.com/FirstianB101)


## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [BYOL](https://arxiv.org/abs/2006.07733)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* Thanks to [FirstianB101](https://github.com/FirstianB101), 모델을 application에 적용할 수 있도록 도움을 주셔서 감사합니다.
