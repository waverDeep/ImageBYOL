import torch


def text_pretext(config, pretext_model, pretext_dataloader, writer, epoch):
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