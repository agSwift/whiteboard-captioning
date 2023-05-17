import copy
from enum import Enum
from pathlib import Path
import numpy as np
import numpy.typing as npt

from pyctcdecode import build_ctcdecoder
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from jiwer import cer, wer
import wandb

import extraction
import dataset
import rnn

wandb.login()

INDEX_TO_CHAR = {index: char for char, index in dataset.CHAR_TO_INDEX.items()}
INDEX_TO_CHAR[0] = "_"  # Epsilon character for the CTC loss.

# Hyperparameters.
BATCH_SIZE = 64
NUM_EPOCHS = 200
HIDDEN_SIZE = 256
NUM_CLASSES = len(dataset.CHAR_TO_INDEX) + 1  # +1 for the epsilon character.
NUM_LAYERS = 5
DROPOUT_RATE = 0.3
LEARNING_RATE = 3e-4
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECODER = build_ctcdecoder([INDEX_TO_CHAR.get(i) for i in range(NUM_CLASSES)])


# Initialize the W&B run.
run = wandb.init(
    # Set the project where this run will be logged.
    project="drawing-gui",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
    },
)


class ModelType(Enum):
    """An enum for the model types."""

    RNN = rnn.RNN
    LSTM = rnn.LSTM
    GRU = rnn.GRU


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-3,
        restore_best_weight: bool = True,
    ):
        """Initializes the EarlyStopping object.

        Args:
            patience (int, optional): The number of epochs to wait for the
                validation loss to improve. Defaults to 7.
            min_delta (float, optional): Minimum difference to consider an
                update in the validation loss. Defaults to 1e-3.
            restore_best_weight (bool, optional): Whether to restore model
                weights from the epoch with the best value of the monitored
                quantity. Defaults to True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weight = restore_best_weight
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """Call the EarlyStopping instance.

        Args:
            model (nn.Module): The model to evaluate.
            val_loss (float): The current validation loss.

        Returns:
            bool: True if the model should be early stopped, False otherwise.
        """
        # Check if the best_loss has been set.
        if self.best_loss is None:
            # Set the best_loss to the current validation loss and store the model.
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        # Check if the current validation loss is lower than the best loss by
        # at least min_delta.
        elif self.best_loss - val_loss > self.min_delta:
            # Update the best loss and reset the counter.
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        # Check if the current validation loss is not lower than the best loss
        # by at least min_delta.
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # If the counter is greater than or equal to patience, then stop training.
            if self.counter >= self.patience:
                self.status = f"Early stopped on {self.counter}"
                # If restore_best_weight is True, restore the model weights from the
                # epoch with the best validation loss
                if self.restore_best_weight:
                    model.load_state_dict(self.best_model.state_dict())
                return True

        # Update the status.
        self.status = f"{self.counter}/{self.patience}"
        print(f"Early Stopping Status: {self.status}")
        return False


def _greedy_decode(indices: npt.NDArray[np.int_]) -> str:
    """Greedy decode character indices to a string by squashing repeated characters.

    Args:
        indices (npt.NDArray[np.int_]): The character indices.

    Returns:
        str: The decoded string.

    Raises:
        TypeError: If indices is not of type numpy.ndarray.
        ValueError: If an index in indices is not found in INDEX_TO_CHAR.
    """
    if not isinstance(indices, np.ndarray):
        raise TypeError(
            f"Expected indices to be of type numpy.ndarray, but got {type(indices)}."
        )

    prev_char = None
    output = []

    for idx in indices:
        if idx not in INDEX_TO_CHAR:
            raise ValueError(f"Index {idx} not found in INDEX_TO_CHAR.")

        curr_char = INDEX_TO_CHAR[idx]

        if curr_char != prev_char:
            output.append(curr_char)

        prev_char = curr_char

    return "".join(output)


def _extract_load_datasets() -> (
    tuple[
        dataset.StrokeBezierDataset,
        dataset.StrokeBezierDataset,
        dataset.StrokeBezierDataset,
        dataset.StrokeBezierDataset,
    ]
):
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
        float: The average training loss.
    """
    # Set the model to training mode.
    model.train()

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
            [len(label) for label in labels_no_padding],
            dtype=torch.int32,
            device=DEVICE,
        )

        # Calculate the CTC loss.
        loss = criterion(logits, labels, input_lengths, target_lengths)
        # Perform backpropagation.
        loss.backward()

        # Update the model parameters.
        optimizer.step()
        # Accumulate the training loss.
        train_loss += loss.item()

    # Calculate the average training loss.
    return train_loss / len(train_loader)


def _validate_epoch(
    *, model: nn.Module, criterion: nn.CTCLoss, val_loader: DataLoader
) -> tuple[float, float, float]:
    """Validates the model for one epoch on the given validation dataset.

    Args:
        model (nn.Module): The model to validate.
        criterion (nn.CTCLoss): The CTC loss function.
        val_loader (DataLoader): The validation data loader.

    Returns:
        tuple[float, float, float]: The average validation loss, CER, and WER for the epoch.
    """
    # Set the model to evaluation mode.
    model.eval()

    val_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0

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
                dtype=torch.int32,
                device=DEVICE,
            )

            # Calculate target_lengths for the current batch.
            labels_no_padding = [label[label != -1] for label in labels]
            labels_to_chars = [
                "".join(
                    [
                        INDEX_TO_CHAR[index]
                        for index in label.detach().cpu().numpy()
                    ]
                )
                for label in labels_no_padding
            ]
            total_samples += len(labels_to_chars)

            greedy_predictions = [
                _greedy_decode(prediction)
                for prediction in logits.argmax(2).detach().cpu().numpy().T
            ]

            for i, (label, prediction) in enumerate(
                zip(labels_to_chars, greedy_predictions)
            ):
                char_error_rate = cer(label, prediction)
                word_error_rate = wer(label, prediction)
                wandb.log(
                    {
                        "Validation - CER": char_error_rate,
                        "Validation - WER": word_error_rate,
                    }
                )

                total_cer += char_error_rate
                total_wer += word_error_rate

                if i == 0:
                    print(f"Label: {label}")
                    print(f"Prediction: {prediction}")
                    print(f"CER: {char_error_rate:.4f}")
                    print(f"WER: {word_error_rate:.4f}")
                    print()

            target_lengths = torch.tensor(
                [len(label) for label in labels_no_padding],
                dtype=torch.int32,
                device=DEVICE,
            )

            # Calculate the CTC loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            # Accumulate the validation loss.
            val_loss += loss.item()

    avg_val_loss = val_loss / total_samples
    avg_cer = total_cer / total_samples
    avg_wer = total_wer / total_samples

    return avg_val_loss, avg_cer, avg_wer


def _test_model(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    test_loader: DataLoader,
) -> tuple[float, float, float]:
    """Tests the model on the given test dataset.

    Args:
        model (nn.Module): The model to test.
        criterion (nn.CTCLoss): The CTC loss function.
        test_loader (DataLoader): The test data loader.

    Returns:
        tuple[float, float, float]: The average test loss, CER, and WER.
    """
    # Set the model to evaluation mode.
    model.eval()

    test_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0

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
                dtype=torch.int32,
                device=DEVICE,
            )

            # Calculate target_lengths for the current batch.
            labels_no_padding = [label[label != -1] for label in labels]
            target_lengths = torch.tensor(
                [len(label) for label in labels_no_padding],
                dtype=torch.int32,
                device=DEVICE,
            )

            labels_to_chars = [
                "".join(
                    [
                        INDEX_TO_CHAR[index]
                        for index in label.detach().cpu().numpy()
                    ]
                )
                for label in labels_no_padding
            ]
            total_samples += len(labels_to_chars)

            greedy_predictions = [
                _greedy_decode(prediction)
                for prediction in logits.argmax(2).detach().cpu().numpy().T
            ]

            for label, prediction in zip(labels_to_chars, greedy_predictions):
                char_error_rate = cer(label, prediction)
                word_error_rate = wer(label, prediction)
                wandb.log(
                    {
                        "Test - CER": char_error_rate,
                        "Test - WER": word_error_rate,
                    }
                )

                total_cer += char_error_rate
                total_wer += word_error_rate

            # Calculate the CTC loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            # Accumulate the test loss.
            test_loss += loss.item()
            wandb.log({"Test - Loss": loss.item()})

    avg_test_loss = test_loss / total_samples
    avg_cer = total_cer / total_samples
    avg_wer = total_wer / total_samples

    return avg_test_loss, avg_cer, avg_wer


def train_model(
    model_type: ModelType,
) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    """Train and evaluate the given model on the training, validation and test datasets.

    Args:
        model_type (ModelType): The type of model to train.

    Returns:
        tuple[nn.Module, list[float], list[float], list[float], list[float]]: The trained model,
            the training loss, the validation loss, the validation CER and the validation WER.
    """
    (
        train_dataset,
        val_1_dataset,
        val_2_dataset,
        test_dataset,
    ) = _extract_load_datasets()

    print(
        f"Train Dataset Size: {len(train_dataset)}\n"
        f"Val 1 Dataset Size: {len(val_1_dataset)}\n"
        f"Val 2 Dataset Size: {len(val_2_dataset)}\n"
        f"Test Dataset Size: {len(test_dataset)}\n"
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

    early_stopping = EarlyStopping(patience=PATIENCE)

    # Set up the loss function and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the training loop.
    for epoch in range(NUM_EPOCHS):
        # Train the model for one epoch.
        train_loss = _train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        wandb.log({"Training - Loss": train_loss})

        # Validate the model on the validation datasets.
        val_1_loss, val_1_cer, val_1_wer = _validate_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_1_loader,
        )
        val_2_loss, val_2_cer, val_2_wer = _validate_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_2_loader,
        )

        # Calculate the average validation loss, CER and WER.
        avg_val_loss = (val_1_loss + val_2_loss) / 2
        avg_val_cer = (val_1_cer + val_2_cer) / 2
        avg_val_wer = (val_1_wer + val_2_wer) / 2
        wandb.log(
            {
                "Validation - Loss Per Epoch": avg_val_loss,
                "Validation - CER Per Epoch": avg_val_cer,
                "Validation - WER Per Epoch": avg_val_wer,
            }
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val CER: {avg_val_cer:.4f}, "
            f"Val WER: {avg_val_wer:.4f}, "
        )

        if early_stopping(model, avg_val_loss):
            print("Early stopping")
            break

    # Evaluate the model on the test dataset.
    test_loss, test_cer, test_wer = _test_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
    )
    print(
        f"Test Loss: {test_loss:.4f}, Test CER: {test_cer:.4f}, Test WER: {test_wer:.4f}"
    )

    # Create a models directory if it doesn't exist.
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    torch.save(
        model.state_dict(), f"models/{model_type.name.lower()}_model.ckpt"
    )

    return model


if __name__ == "__main__":
    if not extraction.EXTRACTED_DATA_PATH.exists():
        extraction.extract_all_data()

    train_model(model_type=ModelType.RNN)
