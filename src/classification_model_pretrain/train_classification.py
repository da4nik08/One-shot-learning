import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from tqdm import tqdm, tqdm_notebook
from src.classification_model_pretrain.metrics import Metrics
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def train_step(model, loss_fn, opt, loader, batch_size):
    loss_per_batches = 0
    elapsed = 0
    start_epoch2 = time.time()
    for i, data in tqdm(enumerate(loader), total=len(train_labels)//batch_size):

        start_epoch = time.time()
        features, labels = data
        features, labels = features.to(device), labels.to(device)
        opt.zero_grad()

        y_pred = model(features)
        loss = loss_fn(y_pred, labels)
        loss.backward()

        opt.step()

        loss_per_batches += loss
        end_epoch = time.time()
        elapsed += (end_epoch - start_epoch)

    print("train = " + str(elapsed))
    print("train + load = " + str(time.time() - start_epoch2))
    return loss_per_batches/(i+1)


def train(model, loss_fn, opt, train_loader, val_loader, batch_size, directory_path, save_treshold=10, epochs=50, model_name='model_name'):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type, module = model.get_info()
    writer = SummaryWriter('/kaggle/working/runs/' + model_name + '{}_{}_{}'.format(model_type, module, timestamp))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    for epoch in range(epochs):
        start_epoch = time.time()
        metrics_valid = Metrics()
        print('EPOCH {}:'.format(epoch + 1))

        model.train()
        avg_loss = train_step(model, loss_fn, opt, train_loader, batch_size)
        model.eval()

        vloss = 0
        counter = 0
        with torch.inference_mode():
            for i, vdata in enumerate(val_loader):
                vfeatures, vlabels = vdata
                vfeatures, vlabels = vfeatures.to(device), vlabels.to(device)

                y_pred = model(vfeatures)
                bloss = loss_fn(y_pred, vlabels)
                vloss += bloss
                y_pred = torch.argmax(y_pred, 1)
                metrics_valid.batch_step(vlabels, y_pred)
                counter = i

        avg_vloss = vloss / (counter + 1)

        scheduler.step()

        valrecall, valprecision, valf1, valacc = metrics_valid.get_metrics()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Accuracy valid {}'.format(valacc))
        print('Recall valid {}'.format(valrecall))
        print('Precision valid {}'.format(valprecision))
        print('Val F1->{}'.format(valf1))

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training': avg_loss, 'Validation': avg_vloss },
                    epoch + 1)
        writer.add_scalars('Validation Metrics',
                    { 'Validation Recall': valrecall, 'Validation Precision': valprecision, 'Validation F1': valf1
                    }, epoch + 1)
        writer.add_scalars('Validation Accuracy',
                    { 'Validation Accuracy': valacc
                    }, epoch + 1)

        if (epoch + 1) % save_treshold == 0:
            model_path = '/kaggle/working/model_svs/' + model_name +'_{}_{}_{}_{}'.format(model_type, module, 
                                                                          timestamp, (epoch + 1))
            torch.save(model.state_dict(), model_path)
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        print("Time per epoch {}s".format(elapsed))