import argparse
import os
import torch
import json
import numpy as np
from datetime import datetime

import src.datasets.dataset as dataset
import src.models.model as model
import src.optimizers.optimizer as optimizer
import src.utils.interface_tensorboard as tensorboard
import src.utils.interface_file_io as file_io
import src.trainers.trainer as trainer
import src.trainers.tester as tester
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def setup_timestamp():
    now = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def save_checkpoint(config, model, optimizer, loss, epoch, mode="best", date=""):
    if not os.path.exists(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])):
        file_io.make_directory(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name']))
    base_directory = os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])
    if mode == "best":
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-best-{}-epoch-{}.pt".format(date, epoch))
    elif mode == "best-ds":
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-best-ds-{}-epoch-{}.pt".format(date, epoch))
    elif mode == 'step':
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-{}-epoch-{}.pt".format(date, epoch))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveBYOL')
    parser.add_argument("--configuration", required=False,
                        default='./config/F10-transfer-ImageBYOL-V3-ResNet18-AdamP.json')
    args = parser.parse_args()
    now = setup_timestamp()
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    if 'pretext' in config['train_type']:
        print(">>> Train Pretext - ImageBYOL <<<")
    print(">> GPU Available: ", torch.cuda.is_available())
    print(">> Config")
    print(config)

    print(">> load dataset ...")

    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, _ = dataset.get_dataloader(config=config, mode='test')

    print("train_loader: {}".format(len(train_loader)))
    print("test_loader: {}".format(len(test_loader)))

    print(">> load pretext model ...")
    pretext_model = model.load_model(config=config, model_name=config["pretext_model_name"],
                                     checkpoint_path=config['pretext_checkpoint'])

    downstream_model = None
    if config['train_type'] == 'transfer_learning':
        downstream_model = model.load_model(config=config, model_name='Downstream', checkpoint_path=None)
        downstream_model.encoder = pretext_model.online_encoder

    if config['train_type'] == 'transfer_learning':
        downstream_model_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
        print("model parameters: {}".format(downstream_model_params))
        print("{}".format(downstream_model))
    else:
        pretext_model_params = sum(p.numel() for p in pretext_model.parameters() if p.requires_grad)
        print("model parameters: {}".format(pretext_model_params))
        print("{}".format(pretext_model))

    print(">> load optimizer ...")
    model_optimizer = None
    if 'pretext' in config['train_type']:
        model_optimizer = optimizer.get_optimizer(pretext_model.parameters(), config)
    elif 'transfer_learning' in config['train_type']:
        model_optimizer = optimizer.get_optimizer(downstream_model.parameters(), config)

    if config['use_cuda']:
        pretext_model = pretext_model.cuda()
        if 'transfer_learning' in config['train_type']:
            downstream_model = downstream_model.cuda()

    print(">> set tensorboard ...")
    writer = tensorboard.set_tensorboard_writer("{}-{}".format(config['tensorboard_writer_name'], now))

    print(">> start train/test ...")
    best_loss = None
    best_epoch = 0
    early_stop = 0
    epoch = config['epoch']
    for count in range(epoch):
        count = count + 1
        print(">> start train ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(train_loader)))
        if 'pretext' in config['train_type']:
            train_loss = trainer.train_pretext(
                config=config, pretext_model=pretext_model, pretext_dataloader=train_loader,
                pretext_optimizer=model_optimizer, writer=writer, epoch=count)
        elif 'transfer_learning' in config['train_type']:
            train_accuracy, train_loss = trainer.train_transfer(config=config, model=downstream_model,
                                                                dataloader=train_loader, optimizer=model_optimizer,
                                                                writer=writer, epoch=count)

        print(">> start test  ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(test_loader)))
        if 'pretext' in config['train_type']:
            test_loss = tester.test_pretext(
                config=config, pretext_model=pretext_model,
                pretext_dataloader=test_loader, writer=writer, epoch=count)
        elif 'transfer_learning' in config['train_type']:
            test_accuracy, test_loss = tester.test_transfer(config=config, model=downstream_model,
                                                              dataloader=train_loader,
                                                              writer=writer, epoch=count)

        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            best_epoch = count

        if 'pretext' in config['train_type']:
            save_checkpoint(config=config, model=pretext_model, optimizer=model_optimizer,
                            loss=test_loss, epoch=best_epoch, mode="best",
                            date='{}'.format(now))
        elif 'transfer_learning' in config['train_type']:
            save_checkpoint(config=config, model=downstream_model, optimizer=model_optimizer,
                            loss=test_loss, epoch=best_epoch, mode="best",
                            date='{}'.format(now))


        print("save pretext checkpoint")

        print("save checkpoint at {} epoch...".format(count))


if __name__ == '__main__':
    main()
