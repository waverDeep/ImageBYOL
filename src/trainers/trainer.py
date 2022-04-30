import src.optimizers.loss as losses
import torch


def train_pretext(config, pretext_model, pretext_dataloader, pretext_optimizer, writer, epoch):
    total_loss = 0.0
    pretext_model.train()
    pretext_model.update_target_network()

    for batch_idx, (image01, image02) in enumerate(pretext_dataloader):
        if config['use_cuda']:
            image01 = image01.cuda()
            image02 = image02.cuda()

        out_loss = pretext_model(image01, image02)

        pretext_model.zero_grad()
        out_loss.backward()
        pretext_optimizer.step()

        writer.add_scalar("Pretext_loss/train_step", out_loss, (epoch - 1) * len(pretext_dataloader) + batch_idx)
        total_loss += len(image01) * out_loss

    total_loss /= len(pretext_dataloader.dataset)  # average loss
    writer.add_scalar('Pretext_loss/train_epoch', total_loss, (epoch - 1))
    return total_loss

def train_transfer(config, model, dataloader, optimizer, writer, epoch):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = losses.set_criterion("CrossEntropyLoss")

    for batch_idx, (input_data, target) in enumerate(dataloader):
        if config['use_cuda']:
            input_data = input_data.cuda()
            target = target.cuda()

        prediction = model(input_data)
        loss = criterion(prediction, target)

        _, predicted = torch.max(prediction.data, 1)
        accuracy = torch.sum(predicted == target) / len(target)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += len(input_data) * loss
        total_accuracy += len(input_data) * accuracy

        if writer is not None:
            writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(dataloader) + batch_idx)
            writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(dataloader) + batch_idx)

    total_loss /= len(dataloader.dataset)  # average loss
    total_accuracy /= len(dataloader.dataset)  # average acc

    if writer is not None:
        writer.add_scalar('Loss/train', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss
