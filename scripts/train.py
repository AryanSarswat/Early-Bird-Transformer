import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, num_classes, optimizer, criterion, scheduler, wandb_log=False, project_name="", experiment_name=""):
        self.model = model
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        
        self.model.to(self.device)
        
        self.wandb_log = wandb_log
        # Initialize wandb
        if self.wandb_log:
            self.run = wandb.init(project=project_name, name=experiment_name)
            wandb.define_metric("Test Accuracy", summary="max")
        
    def train(self, train_loader, test_loader, epochs, save_every=[10, 30, 50, 100, 150, 200, 250, 300]):
        """
        Train the model

        Args:
            train_loader (Dataloader): Train dataloader
            test_loader (Dataloader): Test dataloader
            epochs (int): Number of epochs
            save_every (list, optional): Save interval.
        """
        for epoch in tqdm(range(epochs)):
            train_accuracy, train_loss = self.train_epoch(train_loader)
            test_accuracy, test_loss = self.evaluate(test_loader)
            
            if not self.wandb_log:
                print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
                print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
                  
            if self.wandb_log:
                self.run.log({"Train Accuracy": train_accuracy, "Train Loss": train_loss, "Test Accuracy": test_accuracy, "Test Loss": test_loss, "Learning Rate" : self.scheduler.get_last_lr()[0]})
            
            if save_every is not None and (epoch+1) in save_every:
                torch.save(self.model.state_dict(), f"./model_weights/{self.experiment_name}_epoch_{epoch+1}.pt")
        
        if self.wandb_log:
            self.run.finish()
        
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch

        Args:
            train_loader (Dataloader): train dataloader
        Returns:
            (tuple): (accuracy, loss)
        """
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
            #labels = F.one_hot(data[1], self.num_classes).to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            self.scheduler.step()
            #print(f"Current Learning rate : {self.scheduler.get_last_lr()}")
        
        return correct / total, epoch_loss / len(train_loader)
            
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test set

        Args:
            test_loader (Dataloader): test dataloader

        Returns:
            (tuple): (accuracy, loss)
        """
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)

                # labels = data[1]
                # if not isinstance(data[1], torch.Tensor):
                #     labels = torch.Tensor(data[1])
                # labels = F.one_hot(labels, self.num_classes).to(self.device)
                
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total, loss / len(test_loader)
        
class Trainer_LRP(Trainer):
    def __init__(self, model, num_classes, optimizer, criterion, scheduler, wandb_log=False, project_name="", experiment_name="", lamb=0.1, disable_lrp_loss=True):
        super().__init__(model, num_classes, optimizer, criterion, scheduler, wandb_log, project_name, experiment_name)
        self.lamb = lamb
        self.disable_lrp_loss = disable_lrp_loss
        
    def train(self, train_loader, test_loader, epochs):
        """
        Train the model

        Args:
            train_loader (Dataloader): Train dataloader
            test_loader (Dataloader): Test dataloader
            epochs (int): Number of epochs
            save_every (list, optional): Save interval.
        """
        for epoch in tqdm(range(epochs)):
            train_accuracy, train_loss = self.train_epoch(train_loader)
            test_accuracy, test_loss = self.evaluate(test_loader)
            
            if not self.wandb_log:
                print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
                print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
                  
            if self.wandb_log:
                self.run.log({"Train Accuracy": train_accuracy, "Train Loss": train_loss, "Test Accuracy": test_accuracy, "Test Loss": test_loss, "Learning Rate" : self.scheduler.get_last_lr()[0]})
        
        if self.wandb_log:
            self.run.finish()
            
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch

        Args:
            train_loader (Dataloader): train dataloader
        Returns:
            (tuple): (accuracy, loss)
        """
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
            #labels = F.one_hot(data[1], self.num_classes).to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            if not self.disable_lrp_loss:
                loss += self.lamb * self.model.get_lrp_weights_sum()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            self.scheduler.step()
            #print(f"Current Learning rate : {self.scheduler.get_last_lr()}")
        
        return correct / total, epoch_loss / len(train_loader)
    
class Trainer_Prune(Trainer):
    def __init__(self, model, num_classes, optimizer, criterion, scheduler, wandb_log, project_name, experiment_name):
        super().__init__(model, num_classes, optimizer, criterion, scheduler, wandb_log, project_name, experiment_name)
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
            #labels = F.one_hot(data[1], self.num_classes).to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    # Do not zero out last MLP
                    if "mlp_head" not in name:
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        m.weight.grad.data.mul_(mask)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            self.scheduler.step()
            #print(f"Current Learning rate : {self.scheduler.get_last_lr()}")
        
        return correct / total, epoch_loss / len(train_loader)