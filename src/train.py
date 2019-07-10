import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import classification_report

from .utils.torch import to_var

from . import config

import pickle as pkl


class Trainer(object):
    def __init__(self, model, optimizer, loss_function, num_epochs=10,
                 use_cuda=True, log_interval=10):

        self.model = model

        self.optimizer = optimizer
        self.loss_function = loss_function

        self.num_epochs = num_epochs

        self.use_cuda = use_cuda
        self.log_interval = log_interval
        
        self.trn_loss_list = []
        self.tst_loss_list = []

    def train_epoch(self, train_batches, test_batch, epoch, writer=None):
        self.model.train()  # Depends on using pytorch
        num_batches = train_batches.num_batches

        total_loss = 0
        for batch_index in tqdm(range(num_batches), desc='Batch'):
            self.model.zero_grad()
#             batch = [batch_index]
#             ret_dict = self.model(batch)
            batch = train_batches.__getitem__(batch_index)
            ret_dict = self.model(batch)

            # FIXME: This part depends both on the way the batch is built and
            # on using pytorch. Think of how to avoid this. Maybe by creating
            # a specific MultNLI Trainer Subclass?
            labels = batch['labels']
            labels = to_var(torch.LongTensor(labels), self.use_cuda,
                            requires_grad=False)

            # FIXME: this line assumes that the loss_function expects logits
            # and that ret_dict will contain that key, but what if our problem
            # is not classification?
            batch_loss = self.loss_function(ret_dict['logits'], labels)

            batch_loss.backward()
            self.optimizer.step()
            if writer is not None:
                writer.add_scalar('data/lr', self.optimizer.lr,
                                  self.optimizer.step_num)

            # We ignore batch 0's output for prettier logging
            if batch_index != 0:
                total_loss += batch_loss.item()
                
            print("train loss: ", batch_loss.item())

            if (batch_index % self.log_interval == 0 and batch_index != 0):

                avg_loss = total_loss / self.log_interval
                tqdm.write(f'Epoch: {epoch}, batch: {batch_index}, loss: {avg_loss}')
                if writer is not None:
                    #  FIXME: using the optimizer step as global step is hacky.
                    # Should have a proper global step <2018-06-26 15:18:11, Jorge Balazs>
                    writer.add_scalar('data/train_loss', avg_loss,
                                      self.optimizer.step_num)
                self.trn_loss_list.append(avg_loss)
                torch.save(self.model.state_dict(), f'model/iest_classifier_{epoch}eopch_{batch_index}batch.mdl')
                with open("result/train_loss.lst", 'wb') as f:
                    pkl.dump(self.trn_loss_list, f)
                    
                    
                self.model.eval()
                ret_dict = self.model(test_batch)
                labels = test_batch['labels']
                labels = to_var(torch.LongTensor(labels), self.use_cuda,
                                requires_grad=False)
                
                batch_loss = self.loss_function(ret_dict['logits'], labels)
                self.tst_loss_list.append(batch_loss)
                with open("result/test_loss.lst", 'wb') as f:
                    pkl.dump(self.tst_loss_list, f)
                
                total_loss = 0
                self.model.train()
                

    def evaluate(self, dev_batches, epoch=None, writer=None):
        self.model.eval()

        try:
            #  FIXME: Hacky way of setting elmo in eval mode
            # <2018-06-29 16:41:27, Jorge Balazs>
            self.model.word_encoding_layer.word_encoding_layer._embedder.eval()
        except AttributeError:
            # We can safely ignore the previous line if the model does not use
            # Elmo
            pass

        num_batches = dev_batches.num_batches
        outputs = []
        true_labels = []
        sent_reprs = []
        tqdm.write("Evaluating...")
        for batch_index in range(num_batches):
            print("batch_idx: {}".format(batch_index))
            batch = dev_batches[batch_index]
            out = self.model(batch)

            outputs.append(out['logits'].cpu().data.numpy())
            sent_reprs.append(out['sent_reprs'].cpu().data.numpy())
            true_labels.extend(batch['labels'])

        output = np.vstack(outputs)
        sent_reprs = np.vstack(sent_reprs)
        pred_labels = output.argmax(axis=1)
        true_labels = np.array(true_labels)
        tqdm.write(classification_report(true_labels, pred_labels,
                                         target_names=config.LABELS))
        num_correct = (pred_labels == true_labels).sum()
        num_total = len(pred_labels)
        accuracy = num_correct / num_total
        tqdm.write(f'\nAccuracy: {accuracy:.3f}\n')
        if writer is not None and epoch is not None:
            writer.add_scalar('data/valid_accuracy', accuracy, epoch)

        # Generate prediction list
        pred_labels = pred_labels.tolist()
        pred_labels = [config.ID2LABEL[label] for label in pred_labels]
        ret_dict = {'accuracy': accuracy,
                    'labels': pred_labels,
                    'output': output,
                    'sent_reprs': sent_reprs}
        return ret_dict
    
    
    
def realtime_graph(x, y, title, y_label, x_label='batch iteration', color='r'):
    plt.clf()           # 画面初期化
    line, = plt.plot(x, y, color, label=label)
    line.set_ydata(y)   # y値を更新
    plt.title(title)  
    plt.xlabel(x_label)    
    plt.ylabel(y_label)    
    plt.legend()        
    plt.grid()          
    plt.draw()          
    plt.pause(5)     # 更新時間間隔
