import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from tqdm import tqdm, tqdm_notebook
from src.siamese_model_train.metrics_writer import MetricsWriter
from src.siamese_model_train.smetrics import Metrics
from src.siamese_model_train.triplet_dataset import TripletDataset
from src.siamese_model_train.triplet_loss import TripletLoss
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def train_step(model, loss_fn, opt, loader, batch_size):
    loss_per_batches = 0
    elapsed = 0
    start_epoch2 = time.time()
    all_embeddings = list()
    all_labels = list()
    for i, data in tqdm(enumerate(loader), total=len(train_labels)//(batch_size)):

        start_epoch = time.time()
        img1, img2, labels = data
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        features = torch.cat((img1, img2), dim=0)
        labels = torch.cat((labels, labels), dim=0)
        # concat img & label
        opt.zero_grad()

        y_pred = model(features)
        loss = TripletLoss(loss_func, y_pred, labels, batch_size=batch_size)
        loss.backward()

        opt.step()

        loss_per_batches += loss
        end_epoch = time.time()
        elapsed += (end_epoch - start_epoch)
        all_embeddings.append(y_pred[:batch_size])
        all_labels.append(labels[:batch_size])

    print("train = " + str(elapsed))
    print("train + load = " + str(time.time() - start_epoch2))
    return loss_per_batches/(i+1), torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def train(model, mw, loss_func, opt, train_loader, val_loader, batch_size, epochs=50):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for epoch in range(epochs):
        mw.print_epoch(epoch + 1)
        metrics_valid = Metrics()

        model.train()
        avg_loss, train_emb, train_labels = train_step(model, loss_func, opt, 
                                                       train_loader, batch_size)
        model.eval()

        vloss = 0
        counter = 0
        vclassloss = 0
        with torch.inference_mode():
            for i, vdata in enumerate(val_loader):
                vfeatures1, vfeatures2, vlabels = vdata
                vfeatures1, vfeatures2, vlabels = vfeatures1.to(device), vfeatures2.to(device), vlabels.to(device)
                vfeatures = torch.cat((vfeatures1, vfeatures2), dim=0)
                vlabels = torch.cat((vlabels, vlabels), dim=0)
                
                outputs = model(vfeatures)
                metrics_valid.batch_step(vlabels[:batch_size], outputs[:batch_size])

                bloss = TripletLoss(loss_func, outputs, vlabels, mode='val', batch_size=batch_size)
                vloss += bloss
                counter = i

        avg_vloss = vloss / (counter + 1)
        scheduler.step()

        valrecall, valprecision, valf1, valacc, mAP = metrics_valid.get_metrics(train_emb, train_labels)
        mw.writer_step(avg_loss, avg_vloss, valrecall, valprecision, valf1, valacc, mAP)
        mw.save_model(model)
        mw.print_time()