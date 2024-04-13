import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, optimizer, criterion, wandb_log=False, project_name="", experiment_name=""):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            
        self.model.to(self.device)
        
        # Initialize wandb
        if wandb_log:
            self.run = wandb.init(project=project_name, name=experiment_name)
        
    def train(self, train_loader, test_loader, epochs, save_every=0):
        """
        Train the model

        Args:
            train_loader (Dataloader): Train dataloader
            test_loader (Dataloader): Test dataloader
            epochs (int): Number of epochs
            save_every (int, optional): Save interval. Defaults to 0.
        """
        for epoch in range(epochs):
            train_accuracy, train_loss = self.train_epoch(train_loader)
            test_accuracy, test_loss = self.evaluate(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
            
            if wandb_log:
                self.run.log({"Train Accuracy": train_accuracy, "Train Loss": train_loss, "Test Accuracy": test_accuracy, "Test Loss": test_loss})
            
            if save_every != 0 and (epoch+1) % save_every == 0:
                torch.save(self.model.state_dict(), f"{self.experiment_name}_epoch_{epoch+1}.pth")
        
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
        
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
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
            for i, data in enumerate(tqdm(test_loader)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total, loss / len(test_loader)