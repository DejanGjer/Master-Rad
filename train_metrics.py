import os

import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_loss_history, plot_accuracy_history

class TrainMetrics:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_loss_accuracy = []

    def update(self, train_loss, val_loss, val_accuracy):
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.val_loss_accuracy.append(val_accuracy)

    def save_metrics_to_csv(self, save_dir):
        # check if metrics are empty
        if not self.train_loss_history or not self.val_loss_history or not self.val_loss_accuracy:
            print("No metrics to save.")
            return
        metrics_df = pd.DataFrame({
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'val_accuracy': self.val_loss_accuracy
        })
        save_path = os.path.join(save_dir, f"train_metrics_{self.model_name}.csv")
        metrics_df.to_csv(save_path, index=False)
        print(f"Metrics saved to {save_path}")

    def plot_metrics(self, save_dir):
        # plot loss history
        loss_save_path = os.path.join(save_dir, f"loss_history_{self.model_name}.png")
        plot_loss_history({
            "train_loss": self.train_loss_history,
            "validation_loss": self.val_loss_history
        }, loss_save_path)
        print(f"Loss history plot saved to {loss_save_path}")
        accuracy_save_path = os.path.join(save_dir, f"accuracy_history_{self.model_name}.png")
        plot_accuracy_history(self.val_loss_accuracy, accuracy_save_path)
        print(f"Accuracy history plot saved to {accuracy_save_path}")


class TestMetrics:
    def __init__(self, model_info):
        self.model_info = model_info
        model_names = self.get_active_model_names(model_info)
        self.test_losses = {model_name: 0.0 for model_name in model_names}
        self.test_accuracies = {model_name: 0.0 for model_name in model_names}

    @ staticmethod
    def get_active_model_names(model_info):
        return [model_name for model_name, info in model_info.items() if info["test"]]
    
    def update(self, model_name, test_loss, test_accuracy):
        if model_name in self.test_losses:
            self.test_losses[model_name] += test_loss
            self.test_accuracies[model_name] += test_accuracy
        else:
            raise ValueError(f"Model {model_name} not found in test metrics.")
        
    def save_metrics_to_csv(self, save_dir):
        # check if metrics are empty
        if not self.test_losses or not self.test_accuracies:
            print("No test metrics to save.")
            return
        metrics_df = pd.DataFrame({
            'model_name': list(self.test_losses.keys()),
            'test_loss': list(self.test_losses.values()),
            'test_accuracy': list(self.test_accuracies.values())
        })
        save_path = os.path.join(save_dir, "test_metrics.csv")
        metrics_df.to_csv(save_path, index=False)
        print(f"Test metrics saved to {save_path}")
        