import re
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchvision import models
from typing import Dict, Tuple, Any

class ImageClassifier:
    def __init__(self, arch: str = 'resnet50', hidden_units: int = 512, learning_rate: float = 0.01, no_classes: int = 102, gpu: bool = False):
        """
        Initialize the ImageClassifier with the given parameters.

        Args:
            arch (str): Architecture of the model.
            hidden_units (int): Number of hidden units in the classifier.
            learning_rate (float): Learning rate for training.
            no_classes (int): Number of classes in the dataset.
            gpu (bool): Use GPU for training if available.
        """
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        self.no_classes = no_classes
        self.model = self._initialize_model()
        
    def _initialize_model(self) -> nn.Module:
        """
        Initialize the model based on the specified architecture.

        Returns:
            nn.Module: Initialized model.
        """
        if re.fullmatch(r"efficientnet_(b[0-7]|v2_[sml])|resnet(18|34|50|101|152)", self.arch):
            model = eval(f'models.{self.arch}(weights="DEFAULT", progress=False)')
            for param in model.parameters():
                param.requires_grad = False
            
            if 'efficientnet' in self.arch:
                model.classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(model.classifier[1].in_features, self.hidden_units)),
                    ('relu', nn.ReLU()),
                    ('fc2', nn.Linear(self.hidden_units, self.no_classes)),
                    ('output', nn.LogSoftmax(dim=1))
                ]))
            else:
                model.fc = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(model.fc.in_features, self.no_classes)),
                    ('output', nn.LogSoftmax(dim=1))
                ]))
        else:
            raise ValueError(f"Unsupported Architecture: {self.arch}! Only EfficientNet & ResNet architectures are supported.")
        
        model.to(self.device)
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    def train(self, datasets: Dict[str, Any], dataloaders: Dict[str, torch.utils.data.DataLoader], epochs: int = 10, early_stopping: bool = False, patience: int = 3, monitor: str = 'validation_loss', restore_best_weights: bool = False) -> None:
        """
        Train the model.

        Args:
            datasets (Dict[str, Any]): Dictionary of datasets.
            dataloaders (Dict[str, torch.utils.data.DataLoader]): Dictionary of data loaders.
            epochs (int): Number of epochs for training.
            early_stopping (bool): Enable early stopping.
            patience (int): Number of epochs to wait for improvement before stopping.
            monitor (str): Metric to monitor for early stopping ('validation_loss' or 'validation_accuracy').
            restore_best_weights (bool): Restore model weights from the epoch with the best monitored metric.
        """
        print(f'Starting {self.arch} pre-trained model training for {epochs} epochs using {"GPU" if self.device.type == "cuda" else "CPU"}...')
        self.epochs = epochs
        best_score = None
        best_model_weights = None
        best_epoch = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            accuracy = 0

            tepoch = tqdm(dataloaders['train'], unit="batch", desc=f"Epoch {epoch+1}/{epochs} Training", leave=True)
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                ps = torch.exp(outputs)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy_step = equals.type(torch.FloatTensor).mean().item()
                accuracy += accuracy_step
                
                tepoch.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy_step * 100.})

            validation_running_loss, validation_accuracy = self._validate_model(dataloaders)

            if early_stopping:
                if monitor == 'validation_loss':
                    score = 1 / (validation_running_loss / len(dataloaders['valid']))
                else:
                    score = validation_accuracy * 100. / len(dataloaders['valid'])
                
                if best_score is None or not (score < best_score):
                    best_score = score
                    best_epoch = epoch + 1
                    best_model_weights = self._extract_light_state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'Early stopping patience: {epochs_no_improve}/{patience}')
                    if epochs_no_improve >= patience or (epoch + 1) == epochs:
                        if (epoch + 1) != epochs:
                            print('Early stopping!')
                        self.epochs = epoch + 1
                        if restore_best_weights:
                            print('Restoring best model weights...')
                            self.model.load_state_dict(best_model_weights, strict=False)
                            self.epochs = best_epoch
                        break

        print(f"Training Report - After {epoch + 1} Epochs:")
        print(f"\t- Train loss: {running_loss / len(dataloaders['train']):.4f}.. ")
        print(f"\t- Train accuracy: {accuracy * 100. / len(dataloaders['train']):.2f}%")
        print(f"\t- Validation loss: {validation_running_loss / len(dataloaders['valid']):.4f}")
        print(f"\t- Validation accuracy: {validation_accuracy * 100. / len(dataloaders['valid']):.2f}%")

        self.model.idx_to_class = {v: k for k, v in datasets['train'].class_to_idx.items()}

        print(f'\nTesting the model on testing dataset:')
        testing_running_loss, testing_accuracy = self._validate_model(dataloaders, phase='test')
        print(f"\t- Testing loss: {testing_running_loss / len(dataloaders['test']):.4f}")
        print(f"\t- Testing accuracy: {testing_accuracy * 100. / len(dataloaders['test']):.2f}%")

    def _validate_model(self, dataloaders: Dict[str, torch.utils.data.DataLoader], phase: str = 'valid') -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            dataloaders (Dict[str, torch.utils.data.DataLoader]): Dictionary of data loaders.
            phase (str): Phase of validation ('valid' or 'test').

        Returns:
            Tuple[float, float]: Validation loss and validation accuracy.
        """
        self.model.eval()
        validation_running_loss = 0
        validation_accuracy = 0

        desc = f"\t# Validation" if phase == 'valid' else "\t# Testing"
        vepoch = tqdm(dataloaders[phase], unit="batch", desc=desc, leave=True)
        with torch.no_grad():
            for inputs, labels in vepoch:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                validation_running_loss += loss.item()
                    
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy_step = equals.type(torch.FloatTensor).mean().item()
                validation_accuracy += accuracy_step
                    
                vepoch.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy_step * 100.})
        return validation_running_loss, validation_accuracy

    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        """
        Save the model checkpoint.

        Args:
            save_dir (str): Directory to save the checkpoint.
            epoch (int): Current epoch.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        light_state_dict = self._extract_light_state_dict()

        checkpoint = {
            'epoch': epoch,
            'state_dict': light_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'idx_to_class': self.model.idx_to_class,
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'learning_rate': self.learning_rate
        }
        filename = save_dir + f'/checkpoint_{self.arch}_h_{self.hidden_units}_lr_{self.learning_rate}_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        print(f'Successfully saved checkpoint to {filename}!')

    def _extract_light_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extract a lighter version of the model state dictionary.

        Returns:
            Dict[str, torch.Tensor]: Light state dictionary.
        """
        light_state_dict = OrderedDict(
            {k: v for k, v in self.model.state_dict().items()
             if 'running' in k or 'fc' in k}
        )
        
        return light_state_dict

    @staticmethod
    def load_checkpoint(filepath: str) -> 'ImageClassifier':
        """
        Load a model checkpoint.

        Args:
            filepath (str): Path to the checkpoint file.

        Returns:
            ImageClassifier: Loaded model.
        """
        checkpoint = torch.load(filepath)
        model = ImageClassifier(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['learning_rate'])
        model.model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        model.idx_to_class = checkpoint['idx_to_class']
        print(f'Successfully loaded checkpoint from {filepath}!')
        return model
