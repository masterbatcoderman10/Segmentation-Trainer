# PyTorch Segmentation Model Trainer

This repository provides a comprehensive model trainer for image segmentation. The trainer can be accessed through the `Trainer` class.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Features](#features)
- [Usage](#usage)
- [Improvements](#improvements)
- [Usage](#usage)

## Installation

To utilize the trainer class, you can clone this repository or simply copy the `trainer.py`, and `utils.py` files into your project.

The `utils.py` file holds a class that is used to compute the Dice coefficient, this is used by the `Trainer` when using early stopping or model checkpoint.

```bash
git clone https://github.com/masterbatcoderman10/Segmentation-Trainer.git
```

## Requirements

- Python 3.6+
- PyTorch 1.6+
- Matplotlib
- Numpy
- tqdm


## Features

- **Comprehensive Logging**: Apart from quantitative metrics suchs as the loss after epoch, the trainer also logs various qualitative metrics to provide an in-depth look at the model's performance throughout training.
    - Comparison between the predicted mask and the ground truth mask
    - Displaying the predictions for each class (for multi-class segmentation)
- **Multifaceted**: This model supports training for both binary and multi-class segmentation tasks.
    - Works with various kinds of loss functions.
- **Early Stopping**: The trainer supports early stopping based on the validation loss or the Dice coefficient.
- **Model Checkpoint**: The trainer saves the model's state based on the Dice coefficient or at the end of model training.

## Usage

In order to use the trainer, a few requirements must be met:

- `model`: A PyTorch model that outputs a tensor of the same shape as the ground truth mask.
- `train_dl`: A PyTorch training dataloader that must return a batch of images and corresponding masks.
    - You may use the [Segmentation Dataset](!https://github.com/masterbatcoderman10/Segmentation-Datasets/tree/main) helper classes defined in the mentioned repository. An example of this usage is shown in the tests/trainer_run.ipynb notebook.
- `loss_function`: A loss function that works on 3-D tensors, and returns a single scalar value.
- `optimizer`: A PyTorch optimizer.

In addition to the requirements above, the following requirements may be specified in order to use additional features:

- `scheduler`: A PyTorch learning rate scheduler.
- `patience`: An integer value that specifies the number of epochs to wait before stopping training if the validation loss or Dice coefficient does not improve.

The Trainer can be instantiated as follows:

```python
# Assuming that the model, train_loader, val_loader, loss function, and optimizer have been defined

trainer = Trainer(
    model=model,
    train_dl=train_dl,
    n_classes=n_classes, # Number of classes in the dataset
    epochs=epochs, # Number of epochs to train the model
    loss_function=loss_function,
    optimizer=optimizer,
    scheduler=scheduler, # Optional - Defaults to None
    patience=patience, # Optional - Defaults to None, may be used for early stopping
)
```

Once instantiated, the trainer can be used as follows:

```python
trainer.fit(
    log=True, # Optional - Defaults to False, if True, the trainer will log training process
    validation=True, # Optional - Defaults to False, if True, the trainer will run a validation pass after each epoch, requires a validation dataloader to be passed in.
    valid_dl=valid_dl, # Will be used if validation is set to True
    model_checkpoint=True, # Optional - Defaults to False, if True, the trainer will save the model's state upon improvement in the validation Dice score.
    model_save_path='model.pth', # Optional - Defaults to "./model.pth", if set, the model will be saved to the specified path.
)
```

## Improvements

- Support for external metrics
- Support for logging validation metrics other than the loss