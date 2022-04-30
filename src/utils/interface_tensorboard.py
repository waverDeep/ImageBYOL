# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
import torchvision
import matplotlib.pyplot as plt
import numpy as np
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# console: tensorboard --logdir=runs --bind_all
# tensorboard --logdir=runs --bind_all > /dev/null 2>&1 &


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def inspect_model(writer, model, data):
    writer.add_graph(model, data)


def close_tensorboard_writer(writer):
    writer.close()


def add_image_on_tensorboard(writer, dataloader, desc="dataset"):
    tdataiter = iter(dataloader)
    images, labels = tdataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('{}/images of {}'.format(desc, desc), img_grid)


def add_confusion_matrix(writer, title, desc, step, label_num, targets, predicts):
    labels = np.arange(label_num)
    # print(labels)
    # print(targets)
    # print(predicts)
    output = confusion_matrix(targets, predicts, labels=labels)
    norm_output = output / output.astype(np.float).sum(axis=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(norm_output)
    xaxis = np.arange(len(labels))
    ax1.set_xticks(xaxis)
    ax1.set_yticks(xaxis)
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    writer.add_figure('{}/{}'.format(title, desc), fig, step)
    plt.close()