from enum import Enum
from pathlib import Path
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import extraction
import dataset
import rnn

# Hyperparameters.
BATCH_SIZE = 64
NUM_EPOCHS = 5
HIDDEN_SIZE = 64
NUM_CLASSES = len(dataset.CHAR_TO_INDEX)
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelType(Enum):
    """An enum for the model types."""

    RNN = rnn.RNN
    LSTM = rnn.LSTM
    GRU = rnn.GRU


def _extract_load_datasets() -> tuple[
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
]:
    """Extracts and loads the data from the IAM dataset.
    
    Returns:
        tuple[dataset.StrokeBezierDataset, dataset.StrokeBezierDataset, dataset.StrokeBezierDataset,
        dataset.StrokeBezierDataset]:
            The training, validation, and test datasets.
    """
    # Load the data.
    all_bezier_data = np.load(extraction.EXTRACTED_DATA_PATH)

    # Create the datasets.
    train = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data, dataset_type=dataset.DatasetType.TRAIN
    )
    val_1 = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data, dataset_type=dataset.DatasetType.VAL_1
    )
    val_2 = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data, dataset_type=dataset.DatasetType.VAL_2
    )
    test = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data, dataset_type=dataset.DatasetType.TEST
    )

    return train, val_1, val_2, test


def _create_data_loaders(
    *,
    train_dataset: dataset.StrokeBezierDataset,
    val_1_dataset: dataset.StrokeBezierDataset,
    val_2_dataset: dataset.StrokeBezierDataset,
    test_dataset: dataset.StrokeBezierDataset,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Creates the data loaders for the given datasets.
    
    Args:
        train_dataset (dataset.StrokeBezierDataset): The training dataset.
        val_1_dataset (dataset.StrokeBezierDataset): The first validation dataset.
        val_2_dataset (dataset.StrokeBezierDataset): The second validation dataset.
        test_dataset (dataset.StrokeBezierDataset): The test dataset.
        
    Returns:
        tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
            The training, validation, and test data loaders.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_1_loader = DataLoader(
        val_1_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    val_2_loader = DataLoader(
        val_2_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_1_loader, val_2_loader, test_loader


def _get_max_num_bezier_curves(
    *,
    train_dataset: dataset.StrokeBezierDataset,
    val_1_dataset: dataset.StrokeBezierDataset,
    val_2_dataset: dataset.StrokeBezierDataset,
    test_dataset: dataset.StrokeBezierDataset,
) -> int:
    """Gets the maximum number of bezier curves in a single sample in the given datasets.
    
    Args:
        train_dataset (dataset.StrokeBezierDataset): The training dataset.
        val_1_dataset (dataset.StrokeBezierDataset): The first validation dataset.
        val_2_dataset (dataset.StrokeBezierDataset): The second validation dataset.
        test_dataset (dataset.StrokeBezierDataset): The test dataset.
        
    Returns:
        int: The maximum number of bezier curves in a single sample in the given datasets.
    """
    max_num_bezier_curves = 0
    for dataset in [train_dataset, val_1_dataset, val_2_dataset, test_dataset]:
        max_num_bezier_curves = max(
            max_num_bezier_curves, dataset.max_num_bezier_curves
        )
    return max_num_bezier_curves


def _train_epoch(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    train_dataset: dataset.StrokeBezierDataset,
) -> float:
    """Trains the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        criterion (nn.CTCLoss): The CTC loss function.
        optimizer (optim.Optimizer): The optimizer to use.
        train_loader (DataLoader): The training data loader.
        
    Returns:
        float: The average training loss for the epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize the training loss.
    train_loss = 0.0

    # Iterate through batches in the training dataset.
    for batch_idx, (bezier_curves, labels) in enumerate(train_loader):
        # Move the batch data to the device (CPU or GPU).
        bezier_curves, labels = bezier_curves.to(DEVICE), labels.to(DEVICE)

        # Clear the gradients of the model parameters.
        optimizer.zero_grad()

        # Perform a forward pass through the model.
        logits = model(bezier_curves)
        # Calculate the log probabilities using log softmax.
        log_probs = logits.log_softmax(2).detach().requires_grad_()
        # Create input_lengths tensor for the CTC loss function.
        input_lengths = torch.full(
            size=(logits.size(0),),
            fill_value=logits.size(1),
            dtype=torch.long,
        )

        # Calculate target_lengths for the current batch.
        target_lengths = train_dataset.target_lengths[
            batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE
        ]

        # Calculate the CTC loss.
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        # Perform backpropagation.
        loss.backward()

        # Update the model parameters.
        optimizer.step()
        # Accumulate the training loss.
        train_loss += loss.item()

    # Calculate the average training loss for the epoch.
    train_loss /= len(train_loader)
    return train_loss


def _validate_epoch(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    val_loader: DataLoader,
    val_dataset: DataLoader,
) -> float:
    """Validates the model for one epoch on the given validation dataset.
    
    Args:
        model (nn.Module): The model to validate.
        criterion (nn.CTCLoss): The CTC loss function.
        val_loader (DataLoader): The validation data loader.
        val_dataset (DataLoader): The validation dataset.
        
    Returns:
        float: The average validation loss for the epoch.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Initialize the validation loss.
    val_loss = 0.0

    # Evaluate the model on the validation dataset.
    with torch.no_grad():
        for bezier_curves, labels in val_loader:
            # Move the batch data to the device (CPU or GPU).
            bezier_curves, labels = (
                bezier_curves.to(DEVICE),
                labels.to(DEVICE),
            )

            # Perform a forward pass through the model.
            logits = model(bezier_curves)
            # Calculate the log probabilities using log softmax.
            log_probs = logits.log_softmax(2)
            # Create input_lengths tensor for the CTC loss function.
            input_lengths = torch.full(
                size=(logits.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
            )

            # Calculate target_lengths for the current batch.
            target_lengths = val_dataset.target_lengths[: len(val_loader)]

            # Calculate the CTC loss.
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            # Accumulate the validation loss.
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def _test_model(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    test_loader: DataLoader,
    test_dataset: dataset.StrokeBezierDataset,
) -> float:
    """Tests the model on the given test dataset.
    
    Args:
        model (nn.Module): The model to test.
        criterion (nn.CTCLoss): The CTC loss function.
        test_loader (DataLoader): The test data loader.
        test_dataset (dataset.StrokeBezierDataset): The test dataset.
        
    Returns:
        float: The average test loss.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Initialize the test loss.
    test_loss = 0.0

    # Create a data loader for the test dataset.
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Evaluate the model on the test dataset.
    with torch.no_grad():
        for bezier_curves, labels in test_loader:
            # Move the batch data to the device (CPU or GPU).
            bezier_curves, labels = (
                bezier_curves.to(DEVICE),
                labels.to(DEVICE),
            )

            # Perform a forward pass through the model.
            logits = model(bezier_curves)
            # Calculate the log probabilities using log softmax.
            log_probs = logits.log_softmax(2)
            # Create input_lengths tensor for the CTC loss function.
            input_lengths = torch.full(
                size=(logits.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
            )

            # Calculate target_lengths for the current batch.
            target_lengths = test_dataset.target_lengths[: len(test_loader)]

            # Calculate the CTC loss.
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            # Accumulate the test loss.
            test_loss += loss.item()

    # Calculate the average test loss
    test_loss /= len(test_loader)
    return test_loss


def train_model(model_type: ModelType) -> nn.Module:
    """Train and evaluate the given model on the training, validation and test datasets.
    
    Args:
        model_type (ModelType): The type of model to train.

    Returns:
        nn.Module: The trained model.
    """
    # Extract and load the datasets.
    (
        train_dataset,
        val_1_dataset,
        val_2_dataset,
        test_dataset,
    ) = _extract_load_datasets()

    # Create data loaders for the datasets.
    (
        train_loader,
        val_1_loader,
        val_2_loader,
        test_loader,
    ) = _create_data_loaders(
        train_dataset=train_dataset,
        val_1_dataset=val_1_dataset,
        val_2_dataset=val_2_dataset,
        test_dataset=test_dataset,
    )

    max_num_curves = _get_max_num_bezier_curves(
        train_dataset=train_dataset,
        val_1_dataset=val_1_dataset,
        val_2_dataset=val_2_dataset,
        test_dataset=test_dataset,
    )
    bezier_curve_dimension = train_dataset.all_bezier_curves[0].shape[1]

    model = model_type.value(
        num_bezier_curves=max_num_curves,
        bezier_curve_dimension=bezier_curve_dimension,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        device=DEVICE,
    ).to(DEVICE)

    # Set up the loss function and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the training loop
    for epoch in range(NUM_EPOCHS):
        # Train the model for one epoch.
        train_loss = _train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            train_dataset=train_dataset,
        )
        # Validate the model on the validation datasets.
        val_1_loss = _validate_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_1_loader,
            val_dataset=val_1_dataset,
        )
        val_2_loss = _validate_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_2_loader,
            val_dataset=val_2_dataset,
        )
        # Calculate the average validation loss.
        avg_val_loss = (val_1_loss + val_2_loss) / 2

        # Print the losses for the current epoch.
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val_1 Loss: {val_1_loss:.4f}, "
            f"Val_2 Loss: {val_2_loss:.4f}, "
            f"Avg Val Loss: {avg_val_loss:.4f}"
        )

    # Evaluate the model on the test dataset.
    test_loss = _test_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        test_dataset=test_dataset,
    )
    print(f"Test Loss: {test_loss:.4f}")

    # Create a models directory if it doesn't exist.
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    torch.save(model.state_dict(), f"models/{model.name.lower()}_model.ckpt")

    # Return the trained model.
    return model


if __name__ == "__main__":
    if not extraction.EXTRACTED_DATA_PATH.exists():
        extraction.extract_all_data()

    train_model(model_type=ModelType.LSTM)
