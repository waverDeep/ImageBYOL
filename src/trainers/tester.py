import torch
import src.optimizers.loss as losses
import src.utils.interface_tensorboard as tensorboard


def test_pretext(config, pretext_model, pretext_dataloader, writer, epoch):
    total_loss = 0.0
    pretext_model.eval()

    with torch.no_grad():
        for batch_idx, (image01, image02) in enumerate(pretext_dataloader):
            if config['use_cuda']:
                image01 = image01.cuda()
                image02 = image02.cuda()

            out_loss = pretext_model(image01, image02)

            writer.add_scalar("Pretext-loss/test-step", out_loss, (epoch - 1) * len(pretext_dataloader) + batch_idx)
            total_loss += len(image01) * out_loss

    total_loss /= len(pretext_dataloader.dataset)  # average loss
    writer.add_scalar('Pretext-loss/test-epoch', total_loss, (epoch - 1))
    return total_loss


def test_transfer(config, model, dataloader, writer, epoch):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_target = []
    total_predict = []
    criterion = losses.set_criterion("CrossEntropyLoss")
    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(dataloader):
            if config['use_cuda']:
                input_data = input_data.cuda()
                target = target.cuda()

            prediction = model(input_data)
            loss = criterion(prediction, target)

            _, predicted = torch.max(prediction.data, 1)
            accuracy = torch.sum(predicted == target) / len(target)



            total_loss += len(input_data) * loss
            total_accuracy += len(input_data) * accuracy

            total_target.append(target.cpu())
            total_predict.append(predicted.cpu())

            if writer is not None:
                writer.add_scalar('Loss/test_step', loss, (epoch - 1) * len(dataloader) + batch_idx)
                writer.add_scalar('Accuracy/test_step', accuracy * 100, (epoch - 1) * len(dataloader) + batch_idx)

        total_loss /= len(dataloader.dataset)  # average loss
        total_accuracy /= len(dataloader.dataset)  # average acc

        if writer is not None:
            writer.add_scalar('Loss/test', total_loss, (epoch - 1))
            writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))

        total_target = torch.cat(total_target, dim=0).numpy()
        total_predict = torch.cat(total_predict, dim=0).numpy()

        tensorboard.add_confusion_matrix(writer=writer, title="downstream-test-confusion_matrix", desc="test",
                                         step=(epoch - 1), label_num=config['class_num'],
                                         targets=total_target, predicts=total_predict)
    return total_accuracy, total_loss
