import torch
from src.classification_model_pretrain.custom_dataset import CustomDataset
from models import ResNet18WithSGEFeatureExtractor


def batch_inference(model, dataloader):
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for i, vdata in enumerate(val_loader):
            vfeatures = vdata.to(device)
            y_pred = model(vfeatures)
            embeddings.append(y_pred.cpu())  

        embeddings = torch.cat(embeddings, dim=0)
    return embeddings 