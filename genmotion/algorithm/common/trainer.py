import torch

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
        