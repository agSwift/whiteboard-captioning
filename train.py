"""Used to train the models."""
from pathlib import Path
from enum import Enum
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import isgl
import rnn

# Hyperparameters.
BATCH_SIZE = 64
INPUT_SIZE = 2  # x and y coordinates.
HIDDEN_SIZE = 64
NUM_CLASSES = len(isgl.dataset.ALL_CHARS)
NUM_LAYERS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelType(Enum):
    """An enum for the model types."""

    RNN = rnn.RNN
    LSTM = rnn.LSTM
    GRU = rnn.GRU


def train_model(
    *,
    model_type: ModelType,
    train_dataset: Subset,
    val_dataset: Subset,
    test_dataset: Subset,
):
    """Trains the given model.

    Args:
        model_type (ModelType): The model type to train.
        train_dataset (Subset): The training dataset.
        val_dataset (Subset): The validation dataset.
        test_dataset (Subset): The test dataset.

    Raises:
        ValueError: If the model type is invalid.

    Returns:
        None.
    """
    if not isinstance(model_type, ModelType):
        raise ValueError(
            f"Invalid model type: {model_type}. Must be an instance of {ModelType}."
        )

    print(f"Training model: {model_type.name}")
    model = model_type.value(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )

    # Create data loaders for the training, validation and test sets.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Loss and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model.
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        for i, (points, labels) in enumerate(pbar):
            points, labels = points.to(DEVICE), labels.to(DEVICE)

            # Forward pass.
            outputs = model(points)
            loss = criterion(outputs, labels)

            # Backward and optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # Update the progress bar with the loss values.
                pbar.set_postfix(Loss=f"{loss.item():.4f}")

        # Validate the model.
        with torch.no_grad():
            correct = 0
            total = 0
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)

                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                labels_idx_max = torch.argmax(labels, dim=1)
                correct += (predicted == labels_idx_max).sum().item()

            print(
                f"Validation accuracy of the model on the {total} "
                f"test points: {100 * correct / total:.2f} %"
            )

    # Test the model.
    with torch.no_grad():
        correct = 0
        total = 0
        for points, labels in test_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            outputs = model(points)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels_idx_max = torch.argmax(labels, dim=1)
            correct += (predicted == labels_idx_max).sum().item()

        print(
            f"Test accuracy of the model on the {total} "
            f"test points: {100 * correct / total:.2f} %",
            end="\n\n",
        )

    # Create a models directory if it doesn't exist.
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the model.
    torch.save(
        model.state_dict(), f"models/{model_type.name.lower()}_model.ckpt"
    )


if __name__ == "__main__":
    # Extract the data if it hasn't been extracted yet.
    if not isgl.extraction.EXTRACTED_DATA_PATH.exists():
        isgl.extraction.extract_all_data()

    # Create the dataset.
    stroke_dataset = isgl.dataset.StrokeDataset(
        np.load(str(isgl.extraction.EXTRACTED_DATA_PATH))
    )

    # Split dataset into training, validation and test sets.
    dataset_size = len(stroke_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        stroke_dataset, [train_size, val_size, test_size]
    )

    # Train the RNN model.
    train_model(
        model_type=ModelType.RNN,
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
    )
