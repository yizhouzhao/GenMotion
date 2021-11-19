import torch
import torch.nn as nn
from tqdm.auto import tqdm

import os
import numpy as np

class Trainer(object):
    def __init__(self, dataset, model, opt, device):
        self.opt = opt
        self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)

        # define optimizer
        self.learning_rate = opt.get("learning_rate", 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        # train / test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = 16, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = 16, shuffle=True)
    
    def train(self, epochs, eval_each_epoch = True):
        pass
    
    def evaluate(self, save_best = True):
        pass

    def save(self, path):
        print("saving model to {}".format(path))
        torch.save(self.model.state_dict(), path)
        


class HDM05Trainer(Trainer):
    def __init__(self, dataset, model, opt, device):
        super().__init__(dataset, model, opt, device)

        # define loss function
        self.align_position_criterion = nn.MSELoss()
        self.align_rotation_criterion = nn.MSELoss()

        # define loss weight
        self.position_loss_weight = opt.get("position_loss_weight", 0.1)
        self.rotation_loss_weight = opt.get("rotation_loss_weight", 1.0)

        # save_path
        self.best_loss = 1e6
        self.model_save_path = opt.get("model_save_path")
        

    def train(self, epochs, eval_each_epoch = True, print_every = 20):
        total_step = 0
        #epoch_progress_bar = tqdm(range(epochs))
        for epoch in range(epochs):
            print("training epoch: ", epoch)
            self.model.train()
            training_loss_record = []
            batch_progress_bar = tqdm(self.train_loader)
            for batch in batch_progress_bar:
                total_step += 1
                input_motion, target_motion = batch

                input_motion = input_motion.to(self.device)
                target_motion = target_motion.to(self.device)
                
                # forward
                predict_motion = self.model(input_motion)

                # loss
                position_loss =  self.align_position_criterion(predict_motion[:,:,:3], target_motion[:,:,:3])
                rotation_loss =  self.align_rotation_criterion(predict_motion[:,:,3:], target_motion[:,:,3:])

                total_loss = self.position_loss_weight * position_loss + self.rotation_loss_weight * rotation_loss 

                # backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # record 
                training_loss_record.append(total_loss.item())

                if total_step % print_every == 0:
                    mean_loss = np.mean(training_loss_record)
                    # print("total_loss", mean_loss)
                    batch_progress_bar.set_postfix({
                        "total loss:": "{:.3f}".format(mean_loss),
                        "pos loss:": "{:.3f}".format(position_loss.item()),
                        "rot loss:": "{:.3f}".format(rotation_loss.item()),
                        })
                    training_loss_record.clear()

            if eval_each_epoch:
                self.evaluate()

        if not eval_each_epoch:
            self.evaluate()
    
    def evaluate(self, save_best = True):
        print("evaluate ......")
        self.model.eval()
        eval_loss_record = {
            "total loss": [],
            "position loss": [],
            "rotation loss": [],
            }
        batch_progress_bar = tqdm(self.test_loader)
        for batch in batch_progress_bar:
            input_motion, target_motion = batch

            input_motion = input_motion.to(self.device)
            target_motion = target_motion.to(self.device)
            
            # forward
            predict_motion = self.model(input_motion)

            # loss
            position_loss =  self.align_position_criterion(predict_motion[:,:,:3], target_motion[:,:,:3])
            rotation_loss =  self.align_rotation_criterion(predict_motion[:,:,3:], target_motion[:,:,3:])
            total_loss = self.position_loss_weight * position_loss + self.rotation_loss_weight * rotation_loss

            # record 
            eval_loss_record['total loss'].append(total_loss.item())
            eval_loss_record['position loss'].append(position_loss.item())
            eval_loss_record['rotation loss'].append(rotation_loss.item())

        print("total loss {:.3f}".format(np.mean(eval_loss_record['total loss'])))
        print("position loss {:.3f}".format(np.mean(eval_loss_record['position loss'])))
        print("rotation loss {:.3f}".format(np.mean(eval_loss_record['rotation loss'])))
        
        if save_best:
            best_loss = np.mean(eval_loss_record['total loss'])
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)

                model_path = os.path.join(self.model_save_path, "best.pth")
                self.save(model_path)
            


       


                


                
            

        


    

        