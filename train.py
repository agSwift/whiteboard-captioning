from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import data_extraction
import dataset
import rnn

# Hyperparameters.
BATCH_SIZE = 64
INPUT_SIZE = 2  # x and y coordinates.
HIDDEN_SIZE = 64
NUM_CLASSES = len(dataset.ALL_CHARS)
NUM_LAYERS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model):
    print(f"Training model: {model}")

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
            f"test points: {100 * correct / total:.2f} %"
        )


if __name__ == "__main__":
    # Extract all stroke data and create a dataset.
    data_extraction.extract_all_data()

    stroke_dataset = dataset.StrokeDataset(
        numbers_data=np.load("data/numbers.npz"),
        lowercase_data=np.load("data/lowercase.npz"),
        uppercase_data=np.load("data/uppercase.npz"),
    )

    # Split dataset into training, validation and test sets.
    dataset_size = len(stroke_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        stroke_dataset, [train_size, val_size, test_size]
    )

    # Create the RNN model.
    rnn_model = rnn.RNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )

    # Create the GRU model.
    gru_model = rnn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )

    # Create the LSTM model.
    lstm_model = rnn.LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )

    # Train the models.
    train_model(rnn_model)
    train_model(gru_model)
    train_model(lstm_model)

    # Create a models directory if it doesn't exist.
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the models.
    torch.save(rnn_model.state_dict(), "models/rnn_model.ckpt")
    torch.save(gru_model.state_dict(), "models/gru_model.ckpt")
    torch.save(lstm_model.state_dict(), "models/lstm_model.ckpt")
