

def train_pretext(config, pretext_model, pretext_dataloader, pretext_optimizer, writer, epoch):
    total_loss = 0.0
    pretext_model.train()
    pretext_model.update_target_weight()

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

