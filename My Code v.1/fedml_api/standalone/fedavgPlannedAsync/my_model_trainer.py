import torch
from torch import nn
import numpy as np

#np.random.seed(0)

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML_sat.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        #print('get_model_params = '+str(self.model.cpu().state_dict()))
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        #print(' set_model_params = '+str(self.model.load_state_dict(model_parameters))
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            #optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        #print('train_data = '+str(train_data))
        for epoch in range(args.epochs):

            #train_data = self.shuffle_data_each_Epoch(train_data)

            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))


    def shuffle_data_each_Epoch(self,train_data):

      shuff_combined_data = []
      temp = []

      for g in range(len(train_data)):
       temp.append(train_data[g][0])

      x=torch.cat((temp),0)
      #print(x)
      index = torch.randperm(x.shape[0])  #Create the indexes of size of x for shuffling the data
      #print('index = '+str(index))
      #print(indexes)
      #print(v)
      shuff_combined_data=[]
      for i in range(2):
        temp=[]
        for g in range(len(train_data)):
          temp.append(train_data[g][i])
        x = torch.cat((temp),0) # combine the data
        x = x[index]  # Shuffle the x data based on the index
        shuff_combined_data.append(x)

        #print(v)


      batch_size = len(train_data[0][0])
      iteration  = len(train_data)
      train_data = []   #Create the batch of the shuffled train_data

      for i in range(iteration):
            batched_x = shuff_combined_data[0][i*batch_size:i*batch_size + batch_size]
            batched_y = shuff_combined_data[1][i*batch_size:i*batch_size + batch_size]
            train_data.append((batched_x, batched_y))

      return train_data









    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction='sum').to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics['test_total'] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics['test_total'] += target.size(0) * target.size(1)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
