import torch
import torch.nn as nn
from tqdm.auto import tqdm

import os
import numpy as np

from ..common.trainer import Trainer


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
            for batch_data in batch_progress_bar:
                total_step += 1
                batch_data = batch_data.transpose(0, 1) # [B T N] -> [T B N]
                pad_data = torch.ones_like(batch_data)

                batch_data = batch_data.to(self.device)
                pad_data = pad_data.to(self.device)
                
                # forward
                kld_loss, mse_loss, _, _ = self.model(batch_data, pad_data)
                kld_loss = kld_loss / len(pad_data)
                mse_loss = mse_loss / len(pad_data)

                total_loss = kld_loss + mse_loss

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
                        "kld loss:": "{:.3f}".format(kld_loss.item()),
                        "mse loss:": "{:.3f}".format(mse_loss.item()),
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
            "kld loss": [],
            "mse loss": [],
            }
        batch_progress_bar = tqdm(self.test_loader)
        for batch_data in batch_progress_bar:
            pad_data = torch.ones_like(batch_data)

            batch_data = batch_data.to(self.device)
            pad_data = pad_data.to(self.device)
            
            # forward
            kld_loss, mse_loss, _, _ = self.model(batch_data, pad_data)
            kld_loss = kld_loss / len(pad_data)
            mse_loss = mse_loss / len(pad_data)

            total_loss = kld_loss + mse_loss

            # record 
            eval_loss_record['total loss'].append(total_loss.item())
            eval_loss_record['kld loss'].append(kld_loss.item())
            eval_loss_record['mse loss'].append(mse_loss.item())

        print("total loss {:.3f}".format(np.mean(eval_loss_record['total loss'])))
        print("position loss {:.3f}".format(np.mean(eval_loss_record['kld loss'])))
        print("rotation loss {:.3f}".format(np.mean(eval_loss_record['mse loss'])))
        
        if save_best:
            best_loss = np.mean(eval_loss_record['total loss'])
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)

                model_path = os.path.join(self.model_save_path, "best.pth")
                self.save(model_path)
            


       


                


                
            

        


    

        