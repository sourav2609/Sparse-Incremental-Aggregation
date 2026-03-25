import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import copy
from torch.optim import lr_scheduler
#from fedml_api.standalone.fedavgPlannedAsync_withoutaug.fedavgPlannedAsync_withoutaug_api import global_global
#self.start_visting_satellite_GS_index = global_global()

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML_sat.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):

    #self.global_global = global_global

    def get_model_params(self):
         return self.model.cpu().state_dict()


    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def train(self, train_data, device, args,  w_global_global):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                #print(f"log_probs = {log_probs}, labels = {labels}")
                loss = criterion(log_probs, labels)
                #print(f"loss = {loss}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #print(f"##### param.grad  = {dir(model)}")
            #for param in model.parameters():
            #        print(f" param = {param} , gradients = {param.grad}")

            #logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))


    '''
    def train(self, train_data, device, args,  w_global_global):
        #print('train data [0] before shuffle = ' + str(train_data[0]))
        model_previous = copy.deepcopy(w_global_global)
        #print(model_previous.cuda)
        #print(model_previous.parameters().device)


        #train_data = self.shuffle_data_each_Epoch(train_data)
        #print('train data [0] after shuffle = ' + str(train_data[0]))
        model = self.model
        #print('model = '+str(model))

        model.to(device)

        #model_previous = copy.deepcopy(model)
        #print('Tttttttttttt model_previous = ' + str(model_previous))




        model.train()
        #print('model.train() = '+str(model.train()))

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            #print('self.model.parameters() = '+str(model.parameters()))
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr) #, , weight_decay=args.wd
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)


        #scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        epoch_loss = []

        #print('train_data[0] = '+str(train_data[0]))
        #print('model = ' + str(model.linear.bias))
        #print('ccccccccccccccccc' +str(next(model.parameters()).is_cuda))
        #print('ccccccccccccccccc' +str(model_previous))
        for epoch in range(args.epochs):    #Here, the local ML is performed by the number of local epochs number

            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):

                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                quad_penalty = 0.0
                counter = 0


                for name, param in model.named_parameters():
                        #if batch_idx<=1:
                         #print(f"name = {name}")
                         quad_penalty += F.mse_loss(param, model_previous[name], reduction='sum')
                        #quad_penalty += F.mse_loss(param, param1, reduction='sum')

                loss += (args.wd/2.0)*quad_penalty

                loss.backward()
                for param in model.parameters():
                               print(f" param = {param} , gradients = {param.grad}")

                #print(f"Gradient function for loss = {loss.grad_fn}")
                xx = copy.deepcopy(self.model)


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                batch_loss.append(loss.item())



            epoch_loss.append(sum(batch_loss) / len(batch_loss))
    '''



    def shuffle_data_each_Epoch(self,train_data):

      shuff_combined_data = []
      temp = []
      #print(f"train_data in shuffle = {train_data}")
      #print(f"train_data in shuffle = {dir(train_data)}")
      #print(f"train_data[0][0] = {train_data[0][0]}")
      #print(f"train_data[0] = {train_data[0]}")


      for g in range(len(train_data)):
       temp.append(train_data[g][0])    #temp consists of the feature input data

      x=torch.cat((temp),0) #All data in one tensor

      index = torch.randperm(x.shape[0])  #Create the indexes of size of x for shuffling the data
      #print('index = '+str(index))

      shuff_combined_data=[]
      for i in range(2):
        temp = []
        for g in range(len(train_data)):
          temp.append(train_data[g][i])
        x = torch.cat((temp),0) # combine the data
        print(f"xxxxxxxxxx = {x}")
        x = x[index]  # Shuffle the x data based on the index
        shuff_combined_data.append(x)   # shuff_combined_data is the input and label data that are shuffled and are in two tensors

      batch_size = len(train_data[0][0])
      iteration  = len(train_data)
      train_data = []   #Create the batch of the shuffled train_data

      for i in range(iteration):
            batched_x = shuff_combined_data[0][i*batch_size:i*batch_size + batch_size]
            batched_y = shuff_combined_data[1][i*batch_size:i*batch_size + batch_size]
            train_data.append((batched_x, batched_y))

      return train_data

    def test(self, test_data, device, args):

        #test_data = self.shuffle_data_each_Epoch(test_data)
        #print('test_data after shuffle :' + str(test_data[0]))
        model = self.model
        #print('train data [0] after shuffle = ' + str(test_data[0]))
        #print('model.parameters() = '+str(model.state_dict()))
        #print('model = '+str(model))

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)    #Features data
                #print('x = '+str(x))
                target = target.to(device)  #Labels

                pred = model(x) #Model predict of data
                #print('target = '+str(target))
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
