import os
from pathlib import Path
import shutil
import re
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

from engine.config import Config
from dataset.data_loader import GenderDataset
from engine.decorators import timer

class Engine():
    def __init__(self, config_path) -> None:
        self.config_path = config_path

        # Read from the configure file
        self.configure = Config(config_path=self.config_path)
        cfg_data = self.configure.read_config()
        self.cfg = self.configure.find_simple_keys(cfg_data)

        self.project_root = os.path.join(self.cfg["PROJECT_ROOT"])
        self.project_root = os.path.abspath(self.project_root)

        # Create a variable that stores the state dictionaries
        self.state_dicts = {'MODEL_STATE_DICT': None,
                            'OPTIMIZER_STATE_DICT': None}

        self.update_configs(self.cfg["PRETRAINED_MODEL"])

        self.init_train_folder()
        if self.cfg["MODE"].lower() == "train":
            self.init_tensorboard_logger()
            
            # Data Loaders
            self.train_loader = self.create_dataloader(split="train")
            self.val_loader = self.create_dataloader(split="valid")
        elif self.cfg["MODE"].lower() == "test":
            self.test_loader = self.create_dataloader(split="test")
    
        self.model = self.build_model()
        print(self.model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.print_configs() 

    def print_configs(self):
        print(f"----------------- {self.cfg['MODE']} -----------------")
        for key in self.cfg.keys():
            print(f"{key}:{self.cfg[key]}")
        print(f"------------------------------------------")

    def build_model(self):
        model = self.create_model()
        model_name = model.__class__.__name__ 

        if self.pretrained_model:
            # Change the last layer of the model
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            # model.fc = nn.Linear(model.fc.in_features, 2)

            ckpt = self.load_checkpoint(self.cfg["PRETRAINED_MODEL"])
            model = self.load_model(model, ckpt["MODEL_STATE_DICT"])
            print(f"Model was trained for {ckpt['LAST_EPOCH']} epochs.")
        else:
            if self.cfg["MODE"].lower() == "train":
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
                # model.fc = nn.Linear(model.fc.in_features, 2)
                print(f"Using {model_name} with Imagenet weights")
            elif self.cfg["MODE"].lower() == "test":
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
                ckpt = self.load_checkpoint("models/efficientnet_b1.pth")
                model = self.load_model(model, ckpt["MODEL_STATE_DICT"])
                print("Using EfficientNet-b1 weights")

        if torch.cuda.is_available():
            model.cuda()

        return model

    def create_model(self):
        """
            - The following model is chosen as the base model
            - The Imagenet pretrained weights are loaded
            - Only the last layer dimension is changed to 1
        """
        model = models.efficientnet_b1(pretrained=True)

        return model

    def load_model(self, model, state_dicts=None):
        if state_dicts is not None:
            # Load model weights
            model.load_state_dict(state_dicts)
        
        return model

    def load_checkpoint(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        return checkpoint

    def create_dataloader(self, split="train"):
        mode = self.cfg["MODE"].lower()
        batch_size = self.cfg["BATCH_SIZE"]

        shuffle = False
        if mode == "train":
            shuffle = True

        dataset = GenderDataset(root_dir=f'data/{split.strip()}', split=split)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    def init_train_folder(self):
        """
            - If pretrained path is original model train folder structure, if so use it.
                - If not, create from scratch
        """
        folder_check = True
        if self.pretrained_model:
            pretrained_model_path = self.cfg["PRETRAINED_MODEL"]
            if self.is_valid_weight_file_path(pretrained_model_path):
                abs_model_path = self.merge_with_root_path(pretrained_model_path)

                if not os.path.exists(abs_model_path):
                    os.makedirs(abs_model_path, exist_ok=True)

                # Use the already created experiment folder for training
                self.exp_folder = Path(abs_model_path).parent.parent
                print("Experiment folder ", self.exp_folder)
                folder_check = False

        # Create new train folders
        if folder_check:
            self.exps_root_folder = os.path.join(self.project_root, "exps")
            if os.path.exists(self.exps_root_folder):
                num_exps = len(os.listdir(self.exps_root_folder))
                exp_name = "exp_" + str(num_exps + 1)
                self.exp_folder = os.path.join(self.exps_root_folder, exp_name)            
            else:
                exp_name = "exp_1"
                self.exp_folder = os.path.join(self.exps_root_folder, exp_name)
            
            self.cfg["EXP_NAME"] = exp_name
            
            os.makedirs(self.exp_folder, exist_ok=True)
            os.mkdir(os.path.join(self.exp_folder, "weights"))
            os.mkdir(os.path.join(self.exp_folder, "logs"))

    def merge_with_root_path(self, file_path):
        local_path = file_path[file_path.index('exps'):]
        new_abs_path = os.path.join(self.project_root, local_path)

        return new_abs_path

    def is_valid_weight_file_path(self, weight_file_path):
        """
            A function that checks whether the provided weight file has the original train folder hierarchy
            exps
            |__exp_(ID)
                |__weights
                    |__<any-name>.pth
        """    
        # Normalize the file path to remove any relative path components
        normalized_path = os.path.normpath(weight_file_path)

        # Matches 'exps/exp_{number}/weights/{filename}.pth' at the end of the string
        pattern = r'exps\/exp_\d+\/weights\/[^\/]+\.pth$'

        # Check if the normalized file path ends with the required structure
        if re.search(pattern, normalized_path):
            return True
        else:
            return False

    def update_configs(self, pretrained_path):
        """
            Update the configurations w.r.t pretrained model states
        """
        self.pretrained_model = self.cfg["PRETRAINED_MODEL"] is not None
        if self.pretrained_model:
            print(f"Pretrained model {self.cfg['PRETRAINED_MODEL']} is provided")
            ckpt = self.load_checkpoint(pretrained_path)

            # Load state dictionaries
            self.state_dicts["MODEL_STATE_DICT"] = ckpt["MODEL_STATE_DICT"]

            # Start epoch for resume training
            self.start_epoch = ckpt["LAST_EPOCH"] + 1
        else:
            self.start_epoch = 1
    
    def load_checkpoint(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        return checkpoint

    def init_tensorboard_logger(self):
        # Set up TensorBoard logging
        log_dir = os.path.join(self.exp_folder, "logs", "train_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = SummaryWriter(log_dir)

        log_dir = os.path.join(self.exp_folder, "logs", "val_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.val_writer = SummaryWriter(log_dir)
    
    def get_save_dict(self, epoch, model, optimizer):
        save_dict = {'LAST_EPOCH': epoch,
                    'MODEL_STATE_DICT': model.state_dict(),
                    'OPTIMIZER_STATE_DICT': optimizer.state_dict(),
                    'LR': optimizer.param_groups[0]['lr']}

        return save_dict

    def save_ckpt(self, save_path, epoch, model, optimizer):
        save_dict = self.get_save_dict(epoch, model, optimizer)
        torch.save(save_dict, save_path)

    @timer
    def iterate_one_epoch(self, data_loader, model, loss_func=None, optimizer=None, device="cpu", mode="train"):
        losses = []
        accuracies = []
        predictions = torch.Tensor().to(device)
        confidences = torch.Tensor().to(device)

        # Set the model to the appropriate mode
        if mode == "train":
            model.train()
        else:
            model.eval()
        
        for inputs, labels in tqdm(data_loader, desc=f"{mode}"):
            if mode == "train":
                optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            with torch.set_grad_enabled(mode == "train"):
                outputs = model(inputs)
                output_probs = torch.sigmoid(outputs)

                preds = torch.argmax(output_probs, axis=1)
                accuracy = accuracy_score(labels.cpu(), preds.cpu())
                predictions = torch.cat((predictions, preds), dim=0)
                predicted_confidences = output_probs[torch.arange(output_probs.size(0)), preds]
                confidences = torch.cat((confidences, predicted_confidences), dim=0)

                if mode != "test":
                    # Use the raw logits (outputs) for computing the loss
                    loss = loss_func(outputs, labels)

                    if mode == "train":
                        loss.backward()
                        optimizer.step()

            if mode != "test":
                losses.append(loss.item())
            accuracies.append(accuracy)

        return losses, accuracies, predictions, confidences
    
    @timer
    def train(self):
        # Save the config file
        shutil.copy(self.config_path, self.exp_folder)
        epochs = self.cfg["N_EPOCHS"]

        if self.start_epoch < epochs:    
            print(f"Training starting...")
            if self.start_epoch > 1:
                print(f"Resume training from epoch-{self.start_epoch}")

            min_acc_val = -1
            min_loss_val = 1e5
            for epoch in range(self.start_epoch, epochs+1): 
                current_lr = self.optimizer.param_groups[0]['lr']

                # Iterate 1 epoch for the train dataset
                out, elapsed_time_train = self.iterate_one_epoch(data_loader=self.train_loader,
                                                                 model=self.model, 
                                                                 loss_func=self.loss, 
                                                                 optimizer=self.optimizer, 
                                                                 device=self.device, 
                                                                 mode="train")
                train_losses, train_acc, _, _= out
                avg_loss_train = sum(train_losses)/len(train_losses)
                avg_acc_train = sum(train_acc)/len(train_acc)

                # Iterate 1 epoch for the validation dataset
                out, elapsed_time_val = self.iterate_one_epoch(data_loader=self.val_loader, 
                                                               model=self.model, 
                                                               loss_func=self.loss, 
                                                               optimizer=self.optimizer, 
                                                               device=self.device, 
                                                               mode="val")
                val_losses, val_acc, _, _= out
                avg_loss_val = sum(val_losses)/len(val_losses)
                avg_acc_val = sum(val_acc)/len(val_acc)

                # Save results to Tensorboard
                self.train_writer.add_scalar('Train/epoch_loss', avg_loss_train, epoch)
                self.train_writer.add_scalar('Train/epoch_acc', avg_acc_train, epoch)
                self.val_writer.add_scalar('Validation/epoch_loss', avg_loss_val, epoch)
                self.val_writer.add_scalar('Validation/epoch_acc', avg_acc_val, epoch)
                
                # Update the learning rate scheduler
                self.scheduler.step(avg_loss_val)

                # Save the epoch weights
                last_epoch_model_save_path = os.path.join(self.exp_folder, "weights", "last_epoch.pth")
                self.save_ckpt(last_epoch_model_save_path, epoch, self.model, self.optimizer)

                if (avg_acc_val > min_acc_val) and (avg_loss_val < min_loss_val):
                    # Save the model with best validation accuracy
                    print("Saving the best model...")
                    best_model_save_path = os.path.join(self.exp_folder, "weights", "best.pth")
                    self.save_ckpt(best_model_save_path, epoch, self.model, self.optimizer)

                    min_acc_val = avg_acc_val
                    min_loss_val = avg_loss_val

                print(f"Epoch: {epoch}/{epochs}\t\
                    lr: {current_lr}\t\
                    train-time: {elapsed_time_train:.2f} sec\t\
                    train loss: {avg_loss_train:.4f}\t\
                    train acc: {avg_acc_train:.4f}\t\
                    val-time: {elapsed_time_val:.2f}\t\
                    val loss: {avg_loss_val:.4f}\t\
                    val acc: {avg_acc_val:.4f}\t\n")
    
    @timer
    def test(self):
        outputs, _ = self.iterate_one_epoch(data_loader=self.test_loader, 
                                                        model=self.model,  
                                                        device=self.device, 
                                                        mode="test")
        _, _, predicted_labels, confidences = outputs
        predicted_labels = predicted_labels.cpu()
        confidences = confidences.cpu()

        # Get GT data
        all_labels = [labels for _, labels in self.test_loader]
        true_labels = torch.cat(all_labels).numpy()

        accuracy = accuracy_score(true_labels, predicted_labels)
        print("Accuracy: ", accuracy)

        precision = precision_score(true_labels, predicted_labels)
        print("Precision: ", precision)

        precisions, recalls, _ = precision_recall_curve(true_labels, confidences)

        plt.figure()
        plt.plot(recalls, precisions, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.exp_folder, "precision_recall_curve.jpg"))

        recall = recall_score(true_labels, predicted_labels)
        print("Recall: ", recall)

        f1 = f1_score(true_labels, predicted_labels)
        print("F1-Score: ", f1)

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix: \n", conf_matrix)

        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.exp_folder, "confusion_matrix.jpg"))

        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure()
        bars = plt.bar(names, values)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics')

        # Adding values on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

        # Save the plot as an image file
        plt.savefig(os.path.join(self.exp_folder, "acc_precision_recall_f1.jpg"))