import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import Dice
import torch.nn as nn
import time
import math

class Trainer:

    def __init__(self, model, train_dl, n_classes, epochs, loss_function, optimizer, scheduler=None, patience=None):

        self.model = model
        self.train_dl = train_dl
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler 
        #Initialize the early stopping mechanism
        self.dice = Dice(n_classes)
        if patience is not None:
            self.patience = patience
            self.best = float('inf') * -1
            self.counter = 0
            self.es = True
        else:
            self.es = False

        self.n_classes = n_classes
        self.multi = True if n_classes > 1 else False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def early_stop(self, metric, mode=0):
        
        #mode=0 is for minimum, mode=1 is for maximum
        
        if mode:
            if metric > self.best:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric < self.best:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter > self.patience:
            return -1
        else:
            return 0

    
    def get_mins(self, seconds):
        
        """This function converts seconds to minutes and seconds"""

        return f"{math.floor(seconds // 60)} mins : {math.floor(seconds % 60)} seconds"
    
    def main_step(self, img_batch, target_batch):

        #Zeroing out previous gradients
        self.optimizer.zero_grad()

        #Make model prediction
        pred = self.model(img_batch)
        
        target_batch = target_batch.float().to(self.device)
            
        #Compute Loss
        loss = self.loss_function(pred, target_batch)
        #Calculate gradients through backpropagation
        loss.backward()            
        #Update the model parameters
        self.optimizer.step()
        
        return loss, pred
    
    def eval_step(self,img_batch, target_batch):
    
        #Assumes the model is already put into evaluation mode
        val_pred = self.model(img_batch)
            
        target_batch = target_batch.float().to(self.device)

        #Compute Loss
        loss = self.loss_function(val_pred, target_batch)
        
        return loss, val_pred
    
    def plot_sample_prediction(self, img_batch, target_batch, pred_batch, ix, background=False):
    
        """This function plots along with a sample image and annotation, the prediction for the sample image
           background : is background included in the annotations
        """
        assert self.n_classes > 0
        multi = False if self.n_classes == 1 else True
        cmap = 'rainbow' if multi else 'gray'
        #Initializing the softmax and the sigmoid
        softmax = nn.Softmax(dim=0)
        sigmoid = nn.Sigmoid()
        
        fig, ax = plt.subplots(1,4, figsize=(8,4))

        #Getting an image and reshaping it
        test_img = img_batch[ix]
        #Removing the image from the gpu and the computation graph
        n_img = test_img.to('cpu').detach().numpy()
        n_img = np.transpose(n_img, (1,2,0))
        H = n_img.shape[0]
        W = n_img.shape[1]
        
        #Getting the corresponding annotation
        #For multi-class it will be (C,H,W)
        #Removing the annotation from the gpu and computation graph
        test_ann = target_batch[ix]
        n_ann = test_ann.to('cpu').detach().numpy()
        if multi:
        #Making channels last for a single annotation
            n_ann = np.rollaxis(n_ann, 0, 3)
        
        #Getting the corresponding prediction
        #For multi-class it will be (C,H,W)
        test_pred = pred_batch[ix]
        if multi:
            #applying the softmax function to the prediction since they are just scores
            test_pred = softmax(test_pred)
        else:
            #applying the sigmoid function to the prediction
            test_pred = sigmoid(test_pred)
        
        #Thresholded prediction
        threshold = torch.nn.Threshold(0.75, 0)
        test_pred_clamped = threshold(test_pred)
        #For multiclass it will be (C,W,H)
        n_pred = test_pred.to('cpu').detach().numpy()
        #n_pred = np.rollaxis(n_pred, 0, 3)
        
        #For multiclass it will be (C,W,H)
        n_pred_clamped = test_pred_clamped.to("cpu").detach().numpy()
        #n_pred_clamped = np.rollaxis(n_pred_clamped, 0, 3)
        
        
        #Creating masks that can be plotted
        if (not background) and multi:
            #This step creates a mask for the background and concatenates it to the front of the annotation and prediction
            #Then the argmax operation is performed to obtain a matrix which can be plotted
            
            bg = np.full((H,W, 1), 0.1)
            n_ann = np.concatenate([bg, n_ann], axis=-1)
            n_ann = np.argmax(n_ann, axis=-1)

            n_pred = np.concatenate([bg, n_pred], axis=0)
            n_pred = np.argmax(n_pred, axis=0)
            
            n_pred_clamped = np.concatenate([bg, n_pred_clamped], axis=0)
            n_pred_clamped = np.argmax(n_pred_clamped, axis=0)
            
            
        elif background and multi:
            
            n_ann = np.argmax(n_ann, axis=-1)
            n_pred = np.argmax(n_pred, axis=0)
            n_pred_clamped = np.argmax(n_pred_clamped, axis=0)
        else:
            
            #Squeeze the annotation and the prediction since imshow expects 3 channels or just a matrix
            n_ann = np.squeeze(n_ann)
            n_pred = np.squeeze(n_pred) 
            n_pred_clamped = np.squeeze(n_pred_clamped)
        

        #Plotting the image
        ax[0].imshow(n_img)
        ax[0].axis("off")
        #Plotting the annotation
        ax[1].imshow(n_ann, cmap=cmap)
        ax[1].axis("off")
        #Plotting the prediction
        ax[2].imshow(n_pred, cmap=cmap)
        ax[2].axis("off")
        #Plotting a thresholded prediction
        ax[3].imshow(n_pred_clamped, cmap=cmap)
        ax[3].axis("off")
        plt.show()

    def plot_class_activations(self, target_batch, pred_batch):
    
        """This function plots the individual class activations given a prediction image. 
        This function assumes that the channels are first.
        This function also assumes that no softmax has been applied
        """
        
        #Reminder -- Change the test pred to soft pred and uncomment the declaration
        
        softmax = nn.Softmax(dim=0)
        
        test_pred = pred_batch[0].detach().to("cpu")
        test_ann = target_batch[0].detach().to("cpu")
        
        soft_pred = softmax(test_pred)
        soft_pred = soft_pred.detach().to("cpu").numpy()
        
        fig, ax = plt.subplots(1, self.n_classes, figsize=(self.n_classes*2, self.n_classes))
        
        for i in range(self.n_classes):
            
            ax[i].imshow(test_ann[i], cmap='gray')
            ax[i].axis("off")
        
        fig, ax2 = plt.subplots(1, self.n_classes, figsize=(self.n_classes*2,self.n_classes))
        
        for i in range(self.n_classes):

            ax2[i].imshow(soft_pred[i], cmap='gray')
            ax2[i].axis("off")
        plt.show()
        
    def accuracy(self, predictions, target):
        """
        Computes the accuracy of the predictions against the true labels.
        """
        with torch.no_grad():
            pred = f.softmax(predictions, dim=1)
            pred_class = torch.argmax(pred, dim=1)
            target_class = torch.argmax(target, dim=1)
            correct = pred_class.eq(target_class).sum().item()
            acc = correct / len(target)
            return acc
        
    def plot(self, e, training_losses, validation_losses):
        
        # Create a list of the epoch numbers
        epochs = range(1, e + 2)

        # Plot the training loss
        plt.plot(epochs, training_losses, 'r-', label='Training Loss', linewidth=2)

        # Plot the validation loss
        plt.plot(epochs, validation_losses, 'b--', label='Validation Loss', linewidth=2)

        # Set the title and labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()
    
    def fit(self, log=True, validation=False, valid_dl=None, model_checkpoint=True, model_save_path="./model.pth"):
        """
        Trains the segmentation model.

        Args:
            log (bool, optional): Whether to log the training progress. Defaults to True.
            validation (bool, optional): Whether to perform validation during training. Defaults to False.
            valid_dl (torch.utils.data.DataLoader, optional): Validation data loader. Required if validation is enabled. Defaults to None.
            model_checkpoint (bool, optional): Whether to save the best model based on validation loss. Defaults to True.
            model_save_path (str, optional): Path to save the trained model. Defaults to "./model.pth".

        Raises:
            AssertionError: If validation is enabled but no validation data loader is provided or if the validation data loader is not a PyTorch DataLoader object.

        Returns:
            None
        """
        training_losses = []
        validation_losses = []
        best_val_loss = float('inf') * -1.0

        if validation:
            assert valid_dl is not None, "Validation is enabled but no validation data loader is provided"
            assert isinstance(valid_dl, torch.utils.data.DataLoader), "Validation data loader is not a PyTorch DataLoader object"
        
        for e in range(self.epochs):
            print(f"Starting epoch : {e+1} -------------------------------------------------------------------")
            elapsed_time = 0
            st = time.time()
            loss_value = 0
            
            #Indicates start of batch
            start = True
            start_2 = True
            total_batches = 0
                        
            #Training Loop
            self.model.train()
            for img_batch, annotation_batch in self.train_dl:
                
                total_batches += 1
                #Putting the images and annotations on the self.device
                img_batch = img_batch.to(self.device)
                #Obtaining the loss and the predictions for current batch - This is multiclass classification
                    
                loss, pred = self.main_step(img_batch, annotation_batch)
                
            
                #Check for the start of the batch to visualize a prediction
                if start:
                    self.plot_sample_prediction(img_batch, annotation_batch, pred, 0, background=True)
                    #Indicate that next batch is not start of epoch
                    if self.multi:
                        print(f"Plotting Activations")
                        self.plot_class_activations(annotation_batch.to(self.device), pred)
                    start = False
                
                #Updating loss by adding loss for current batch  
                loss_value += loss.item()
                if start_2:
                    print(f"The loss on the first batch is : {loss_value}")
                    start_2 = False
            
            #If logging is enabled print total loss value for the epoch divided by batch size
            if log:
                loss_for_epoch = round(loss_value / total_batches, 3)
                training_losses.append(loss_for_epoch)
                print(f"Loss at epoch : {e+1} : {loss_for_epoch}")

            
            #Validation Loop
            ######################################################################################################################################

            if validation and valid_dl is not None:

                print("Running Validation Step")
                ######### Validation step ############
                val_loss = 0
                val_dice_score = 0
                val_start = True
                val_start_2 = True
                val_batches = 0
                with torch.no_grad():
                    self.model.eval()

                    for img_batch, annotation_batch in valid_dl:
                        
                        val_batches += 1
                        val_img_batch = img_batch.to(self.device)
                        
                        valid_loss, val_pred = self.eval_step(val_img_batch, annotation_batch)
                        #Compute the dice metric
                        val_dice_score += self.dice(val_pred, annotation_batch.float().to(self.device)).item()
                        
                        if val_start:
                            self.plot_sample_prediction(val_img_batch, annotation_batch, val_pred, 0, background=True)
                            val_start = False

                        val_loss += valid_loss.item()

                        if val_start_2:
                            print(f"The loss on the first batch for validation is : {val_loss}")
                            val_start_2 = False


                #If logging is enabled print total loss value for the epoch divided by batch size
                if log:
                    val_loss_for_epoch = round(val_loss / val_batches, 3)
                    val_dice_for_epoch = round(val_dice_score / val_batches, 3)
                    validation_losses.append(val_loss_for_epoch)
                    print(f"Validation Loss at epoch : {e+1} : {val_loss_for_epoch}")
                    print(f"Validation Dice Score at epoch : {e+1} : {val_dice_for_epoch}")
                
                # Saving the best version of the model
                if val_dice_for_epoch > best_val_loss and model_checkpoint:
                    best_val_loss = val_dice_for_epoch
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Saved model at val dice : {best_val_loss}")
                
                
                #Early Stopping
                if self.es:
                    #Early stop if dice validation is less than the max
                    #Return value of the early stop
                    rv = self.early_stop(val_dice_for_epoch, mode=1)
                    if rv == -1:
                        print(f"Early Stopping kicked in : Stopping at epoch {e+1}")
                        break
                
                #Modifying learning rate
                if self.scheduler is not None:
                    self.scheduler.step(val_dice_for_epoch)

                
            #End of Epoch -----------------------------------------------------------------------------------------------------------------------
            #Calculate the end time and log
            et = time.time()
            elapsed_time = et - st
            print(f"Epoch : {e+1} took {self.get_mins(elapsed_time)}")
            print("\n")
                
            ######### End of validation step #######
            print("------------------------------------------------------------------------------------------")
            print("\n")
            print("\n")
            print("\n")
        
        #Used to save the model at the end of training if model checkpoint wasn't enabled
        if not model_checkpoint:
            torch.save(self.model.state_dict(), model_save_path)

        # Create a list of the epoch numbers
        epochs = range(1, e + 2)

        # Plot the training loss
        plt.plot(epochs, training_losses, 'r-', label='Training Loss', linewidth=2)

        # Plot the validation loss
        if validation:
            plt.plot(epochs, validation_losses, 'b--', label='Validation Loss', linewidth=2)
            # Set the title and labels
            plt.title('Training and Validation Loss')
        else:
            plt.title("Training Loss")
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Add a legend
        plt.legend()
        # Show the plot
        file_name = model_save_path[:-4] + ".png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    
    