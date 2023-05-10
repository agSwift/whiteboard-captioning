from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import extraction
import dataset
import rnn

INDEX_TO_CHAR = {
    index: char for char, index in dataset.CHAR_TO_INDEX.items()
}

# Hyperparameters.
BATCH_SIZE = 32
NUM_EPOCHS = 50
HIDDEN_SIZE = 512
NUM_CLASSES = len(dataset.CHAR_TO_INDEX)
NUM_LAYERS = 1
DROPOUT_RATE = 0.5
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


def _train_epoch(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
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
    for bezier_curves, labels in train_loader:
        # Move the batch data to the device (CPU or GPU).
        bezier_curves, labels = bezier_curves.to(DEVICE), labels.to(DEVICE)

        # Remove the extra dimension from the bezier curves.
        bezier_curves = bezier_curves.squeeze(-2)
        # (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim).
        bezier_curves = bezier_curves.permute(1, 0, 2)

        # Clear the gradients of the model parameters.
        optimizer.zero_grad()

        # Perform a forward pass through the model.
        logits = model(bezier_curves)
        # Create input_lengths tensor for the CTC loss function.
        actual_batch_size = bezier_curves.size(1)

        # Create input_lengths tensor for the CTC loss function.
        input_lengths = torch.full(
            size=(actual_batch_size,),
            fill_value=logits.size(0),
            dtype=torch.int32,
            device=DEVICE,
        )

        # Calculate target_lengths for the current batch.
        labels_no_padding = [label[label != -1] for label in labels]
        target_lengths = torch.tensor(
            [len(label) for label in labels_no_padding], dtype=torch.int32, device=DEVICE,
        )

        # Calculate the CTC loss.
        loss = criterion(logits, labels, input_lengths, target_lengths)
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
    *, model: nn.Module, criterion: nn.CTCLoss, val_loader: DataLoader
) -> float:
    """Validates the model for one epoch on the given validation dataset.
    
    Args:
        model (nn.Module): The model to validate.
        criterion (nn.CTCLoss): The CTC loss function.
        val_loader (DataLoader): The validation data loader.
        
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
            # Remove the extra dimension from the bezier curves.
            bezier_curves = bezier_curves.squeeze(-2)
            # (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim).
            bezier_curves = bezier_curves.permute(1, 0, 2)
            # Move the batch data to the device (CPU or GPU).
            bezier_curves, labels = (
                bezier_curves.to(DEVICE),
                labels.to(DEVICE),
            )

            # Perform a forward pass through the model.
            logits = model(bezier_curves)

            actual_batch_size = bezier_curves.size(1)
            # Create input_lengths tensor for the CTC loss function.
            input_lengths = torch.full(
                size=(actual_batch_size,),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=DEVICE,
            )

            # Calculate target_lengths for the current batch.
            labels_no_padding = [label[label != -1] for label in labels]
            predictions = logits.argmax(2).detach().cpu().numpy().T
            # print("predictions shape: ", predictions[0])
            predictions = [
                "".join([INDEX_TO_CHAR[index] for index in prediction if index != 0])
                for prediction in predictions
            ]
            labels_to_chars = [
                "".join([INDEX_TO_CHAR[index] for index in label.detach().cpu().numpy()])
                for label in labels_no_padding
            ]
            print('Prediction: ', predictions[0])
            print('Label: ', labels_to_chars[0])

            target_lengths = torch.tensor(
                [len(label) for label in labels_no_padding], dtype=torch.long, device=DEVICE,
            )

            # Calculate the CTC loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            # Accumulate the validation loss.
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def _test_model(
    *, model: nn.Module, criterion: nn.CTCLoss, test_loader: DataLoader,
) -> float:
    """Tests the model on the given test dataset.
    
    Args:
        model (nn.Module): The model to test.
        criterion (nn.CTCLoss): The CTC loss function.
        test_loader (DataLoader): The test data loader.
        
    Returns:
        float: The average test loss.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Initialize the test loss.
    test_loss = 0.0

    # Evaluate the model on the test dataset.
    with torch.no_grad():
        for bezier_curves, labels in test_loader:
            # Remove the extra dimension from the bezier curves.
            bezier_curves = bezier_curves.squeeze(-2)
            # (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim).
            bezier_curves = bezier_curves.permute(1, 0, 2)

            # Move the batch data to the device (CPU or GPU).
            bezier_curves, labels = (
                bezier_curves.to(DEVICE),
                labels.to(DEVICE),
            )

            # Perform a forward pass through the model.
            logits = model(bezier_curves)

            # Create input_lengths tensor for the CTC loss function.
            actual_batch_size = bezier_curves.size(1)
            input_lengths = torch.full(
                size=(actual_batch_size,),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=DEVICE,
            )

            # Calculate target_lengths for the current batch.
            labels_no_padding = [label[label != -1] for label in labels]
            target_lengths = torch.tensor(
                [len(label) for label in labels_no_padding], dtype=torch.long, device=DEVICE,
            )

            # Calculate the CTC loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            # Accumulate the test loss.
            test_loss += loss.item()

    # Calculate the average test loss
    test_loss /= len(test_loader)
    return test_loss


def train_model(model_type: ModelType) -> tuple[nn.Module, list[float], list[float]]:
    """Train and evaluate the given model on the training, validation and test datasets.
    
    Args:
        model_type (ModelType): The type of model to train.

    Returns:
        tuple[nn.Module, list[float], list[float]]: A tuple containing the trained model,
            the list of validation losses and the list of test losses.
    """
    (
        train_dataset,
        val_1_dataset,
        val_2_dataset,
        test_dataset,
    ) = _extract_load_datasets()

    print(
        f"Train dataset size: {len(train_dataset)}\n"
        f"Validation 1 dataset size: {len(val_1_dataset)}\n"
        f"Validation 2 dataset size: {len(val_2_dataset)}\n"
        f"Test dataset size: {len(test_dataset)}\n"
    )

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

    bezier_curve_dimension = train_dataset.all_bezier_curves[0].shape[-1]

    model = model_type.value(
        bezier_curve_dimension=bezier_curve_dimension,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        device=DEVICE,
    ).to(DEVICE)

    # Set up the loss function and optimizer.
    criterion = nn.CTCLoss(blank=NUM_CLASSES-1, zero_infinity=True, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    # Start the training loop.
    for epoch in range(NUM_EPOCHS):
        # Train the model for one epoch.
        train_loss = _train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_losses.append(train_loss)

        # Validate the model on the validation datasets.
        val_1_loss = _validate_epoch(
            model=model, criterion=criterion, val_loader=val_1_loader,
        )
        val_2_loss = _validate_epoch(
            model=model, criterion=criterion, val_loader=val_2_loader,
        )
        # Calculate the average validation loss.
        avg_val_loss = (val_1_loss + val_2_loss) / 2
        val_losses.append(avg_val_loss)

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
        model=model, criterion=criterion, test_loader=test_loader,
    )
    print(f"Test Loss: {test_loss:.4f}")

    # Create a models directory if it doesn't exist.
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    torch.save(
        model.state_dict(), f"models/{model_type.name.lower()}_model.ckpt"
    )

    # Return the trained model.
    return model, train_losses, val_losses

def plot_losses(train_losses: list[float], val_losses: list[float], model_type: ModelType):
    """Plot the training and validation losses for the given model.
    
    Args:
        train_losses (list[float]): The training losses.
        val_losses (list[float]): The validation losses.
        model_type (ModelType): The type of model.
        
    Returns:
        None
    """
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_type.name} Training and Validation Loss")
    plt.savefig(f"{model_type.name.lower()}_loss_plot.png")
    plt.show()


if __name__ == "__main__":
    if not extraction.EXTRACTED_DATA_PATH.exists():
        extraction.extract_all_data()

    rnn_model, rnn_train_losses, rnn_val_losses = train_model(model_type=ModelType.RNN)
    # lstm_model, lstm_train_losses, lstm_val_losses = train_model(model_type=ModelType.LSTM)
    # gru_model, gru_train_losses, gru_val_losses = train_model(model_type=ModelType.GRU)

    plot_losses(rnn_train_losses, rnn_val_losses, ModelType.RNN)
    # plot_losses(lstm_train_losses, lstm_val_losses, ModelType.LSTM)
    # plot_losses(gru_train_losses, gru_val_losses, ModelType.GRU)
