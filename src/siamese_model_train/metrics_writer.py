import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class MetricsWriter():
    def __init__(self, model, model_name='model_name', save_treshold=10):
        self.loss_val = list()
        self.loss_train = list()
        self.acc_val = list()
        self.mAP = list()
        
        self.epoch = 0
        self.start_epoch = 0

        self.save_treshold = save_treshold
        self.model_name = model_name
        self.model_type, self.module = model.get_info()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('/kaggle/working/runs/' + self.model_name + '{}_{}_{}'.format(self.model_type, 
                                                                                                  self.module, self.timestamp))

    def get_list_metrics(self):
        return self.loss_val, self.loss_train, self.acc_val

    def print_epoch(self, epoch):
        self.start_epoch = time.time()
        self.epoch = epoch
        print('EPOCH {}:'.format(epoch))

    def print_time(self):
        end_epoch = time.time()
        elapsed = end_epoch - self.start_epoch
        print("Time per epoch {}s".format(elapsed))

    def get_best_epoch_by_map(self):
        return self.mAP.index(max(self.mAP))

    def writer_step(self, loss, vloss, recall, precision, f1, acc, mAP):
        print('LOSS train {} valid {}'.format(loss, vloss))
        print('Accuracy valid {}'.format(acc))
        print('Recall valid {}'.format(recall))
        print('Precision valid {}'.format(precision))
        print('Val F1->{}'.format(f1))
        print('Val mAP->{}'.format(mAP))
        
        self.loss_train.append(loss)
        self.loss_val.append(vloss)
        self.acc_val.append(acc)
        self.mAP.append(mAP)

        self.writer.add_scalars('Training vs. Validation Loss',
                    { 'Training': loss, 'Validation': vloss },
                    self.epoch)
        self.writer.add_scalars('Validation Metrics',
                    { 'Validation Recall': recall, 'Validation Precision': precision, 'Validation F1': f1
                    }, self.epoch)
        self.writer.add_scalars('Validation Accuracy',
                    { 'Validation Accuracy': acc
                    }, self.epoch)
        self.writer.add_scalars('Validation mAP',
                    { 'Validation mAP': mAP
                    }, self.epoch)

    def save_model(self, model):
        if (self.epoch) % self.save_treshold == 0:
            model_path = '/kaggle/working/model_svs/' + self.model_name + '_{}_{}_{}_{}'.format(self.model_type, 
                                                                                                self.module, self.timestamp, 
                                                                                                self.epoch)
            torch.save(model.state_dict(), model_path)