"""For training models on the IAM dataset."""
import copy

# import multiprocessing
from enum import Enum
from pathlib import Path
import numpy as np
import numpy.typing as npt

# from pyctcdecode import build_ctcdecoder
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# from sklearn.preprocessing import MinMaxScaler

from iam import extraction, dataset, rnn

# from ctcdecode import CTCBeamDecoder
from jiwer import cer, wer
import wandb

INDEX_TO_CHAR = {index: char for char, index in dataset.CHAR_TO_INDEX.items()}
INDEX_TO_CHAR[0] = "_"  # Epsilon character for CTC loss.

# Hyperparameters.
BATCH_SIZE = 32
NUM_EPOCHS = 300
HIDDEN_SIZE = 256
NUM_CLASSES = len(dataset.CHAR_TO_INDEX) + 1  # +1 for the epsilon character.
BEZIER_CURVE_DEGREE = 5
CROSS_VALIDATION = False
REDUCTION = "mean"
NUM_LAYERS = 3
DROPOUT_RATE = 0.4
BIDIRECTIONAL = True
LEARNING_RATE = 1e-3
PATIENCE = 20
BEAM_WIDTH = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # TODO: Add new decoder or fix CTC decoding.
# DECODER = build_ctcdecoder(
#    labels=[INDEX_TO_CHAR.get(i) for i in range(1, NUM_CLASSES)],
# )

# DECODER = CTCBeamDecoder(
#     [INDEX_TO_CHAR.get(i) for i in range(NUM_CLASSES)],
#     model_path=None,
#     alpha=0,
#     beta=0,
#     cutoff_top_n=100,
#     cutoff_prob=1.0,
#     beam_width=BEAM_WIDTH,
#     num_processes=multiprocessing.cpu_count(),
#     blank_id=0,
#     log_probs_input=True,
# )


class ModelType(Enum):
    """An enum for the model types."""

    RNN = rnn.RNN
    LSTM = rnn.LSTM
    GRU = rnn.GRU


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        *,
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


def get_model_file_name(
    *,
    model_name: str,
    bezier_curve_degree: int,
    num_layers: int,
    bidirectional: bool,
) -> str:
    """Get the file name of the model.

    Args:
        model_name (str): The name of the model.
        bezier_curve_degree (int): The degree of the bezier curve.
        num_layers (int): The number of recurrent layers.
        bidirectional (bool): Whether the model is bidirectional.

    Returns:
        str: The file name of the model.
    """

    model_file_name_template = (
        "{model_name}_"
        + "degree_{bezier_curve_degree}_"
        + "layers_{num_layers}_"
        + "{bidirectional}"
    )

    return model_file_name_template.format(
        model_name=model_name,
        bezier_curve_degree=bezier_curve_degree,
        num_layers=num_layers,
        bidirectional="bidirectional" if bidirectional else "unidirectional",
    )


def greedy_decode(indices: npt.NDArray[np.int_]) -> str:
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

        curr_char = (
            "" if idx == 0 else INDEX_TO_CHAR[idx]
        )  # Ignore the epsilon character.

        if curr_char != prev_char:
            output.append(curr_char)

        prev_char = curr_char

    return "".join(output)


def _extract_load_datasets(
    *,
    bezier_degree: int,
    with_cross_val: bool,
) -> tuple[
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
    dataset.StrokeBezierDataset,
]:
    """Extracts and loads the data from the IAM dataset.

    Args:
        bezier_degree (int): The degree of the bezier curve.
        with_cross_val (bool): Whether to use cross validation or not.

    Returns:
        tuple[dataset.StrokeBezierDataset, dataset.StrokeBezierDataset, dataset.StrokeBezierDataset,
        dataset.StrokeBezierDataset]:
            The training, validation, and test datasets.
    """
    extracted_data_path = extraction.get_extracted_data_file_path(
        with_cross_val=with_cross_val, bezier_degree=bezier_degree
    )

    if not extracted_data_path.exists():
        extraction.extract_all_data(
            with_cross_val=with_cross_val, bezier_degree=bezier_degree
        )

    # Load the data.
    all_bezier_data = np.load(extracted_data_path)

    # TODO: Record metrics when using MinMaxScaler.
    # Fit MinMaxScaler on the training data.
    # scaler = MinMaxScaler()
    # train_bezier_curves = all_bezier_data["train_bezier_curves"].reshape(-1, 10)
    # scaler.fit(train_bezier_curves)

    # Create the datasets.
    train_cross_val = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data,
        dataset_type=extraction.DatasetType.TRAIN_CROSS_VAL,
    )
    train_single_val = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data,
        dataset_type=extraction.DatasetType.TRAIN_SINGLE_VAL,
    )
    val_1 = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data,
        dataset_type=extraction.DatasetType.VAL_1,
    )
    val_2 = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data,
        dataset_type=extraction.DatasetType.VAL_2,
    )
    test = dataset.StrokeBezierDataset(
        all_bezier_data=all_bezier_data,
        dataset_type=extraction.DatasetType.TEST,
    )

    return train_cross_val, train_single_val, val_1, val_2, test


def _create_data_loaders(
    *,
    train_cross_val_dataset: dataset.StrokeBezierDataset,
    train_single_val_dataset: dataset.StrokeBezierDataset,
    val_1_dataset: dataset.StrokeBezierDataset,
    val_2_dataset: dataset.StrokeBezierDataset,
    test_dataset: dataset.StrokeBezierDataset,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Creates the data loaders for the given datasets.

    Args:
        train_cross_val_dataset (dataset.StrokeBezierDataset): The cross validation
            training dataset.
        train_single_val_dataset (dataset.StrokeBezierDataset): The single validation
            training dataset.
        val_1_dataset (dataset.StrokeBezierDataset): The first validation dataset.
        val_2_dataset (dataset.StrokeBezierDataset): The second validation dataset.
        test_dataset (dataset.StrokeBezierDataset): The test dataset.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
            The training, validation, and test data loaders.
    """
    train_cross_val_loader = DataLoader(
        train_cross_val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_single_val_loader = DataLoader(
        train_single_val_dataset, batch_size=BATCH_SIZE, shuffle=True
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
    return (
        train_cross_val_loader,
        train_single_val_loader,
        val_1_loader,
        val_2_loader,
        test_loader,
    )


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
        # Transform from shape (batch_size, seq_len, feature_dim) to
        # (seq_len, batch_size, feature_dim).
        bezier_curves = bezier_curves.permute(1, 0, 2)

        # Clear the gradients of the model parameters.
        optimizer.zero_grad()

        # Perform a forward pass through the model.
        logits = model(bezier_curves)

        # Create an input_lengths tensor for the CTC loss function.
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
        # Calculate the CTC loss and perform backpropagation.
        loss = criterion(logits, labels, input_lengths, target_lengths)
        loss.backward()

        # Update the model parameters and accumulate the training loss.
        optimizer.step()
        train_loss += loss.item()

    # Calculate the average training loss.
    return train_loss / len(train_loader)


def _validate_epoch(
    *, model: nn.Module, criterion: nn.CTCLoss, val_loader: DataLoader
) -> tuple[float, float, float, float, float]:
    """Validates the model for one epoch on the given validation dataset.

    Args:
        model (nn.Module): The model to validate.
        criterion (nn.CTCLoss): The CTC loss function.
        val_loader (DataLoader): The validation data loader.

    Returns:
        tuple[float, float, float, float, float]:
            The average validation loss, greedy CER, greedy WER, beam CER, and beam WER.
    """
    # Set the model to evaluation mode.
    model.eval()

    val_loss = 0.0
    num_samples = 0

    greedy_total_cer = 0.0
    greedy_total_wer = 0.0
    beam_total_cer = 0.0
    beam_total_wer = 0.0

    # Evaluate the model on the validation dataset.
    with torch.no_grad():
        for bezier_curves, labels in val_loader:
            # Remove the extra dimension from the bezier curves.
            bezier_curves = bezier_curves.squeeze(-2)
            # Transform from shape (batch_size, seq_len, feature_dim) to
            # (seq_len, batch_size, feature_dim).
            bezier_curves = bezier_curves.permute(1, 0, 2)
            # Move the batch data to the device (CPU or GPU).
            bezier_curves, labels = (
                bezier_curves.to(DEVICE),
                labels.to(DEVICE),
            )

            # Perform a forward pass through the model.
            logits = model(bezier_curves)

            # Create an input_lengths tensor for the CTC loss function.
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

            # Convert the labels to strings for the CER and WER calculations.
            labels_to_strings = [
                "".join(
                    [
                        INDEX_TO_CHAR[index]
                        for index in label.detach().cpu().numpy()
                    ]
                )
                for label in labels_no_padding
            ]

            # Calculate the CTC loss and accumulate the validation loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            val_loss += loss.item()

            # Get the greedy search decodings for the current batch.
            greedy_predictions = logits.argmax(2).detach().cpu().numpy().T
            greedy_decodings = [
                greedy_decode(prediction) for prediction in greedy_predictions
            ]

            # Get the beam search decodings for the current batch.
            # beam_predictions = logits.detach().cpu().numpy()
            # beam_decodings = [
            #     DECODER.decode(prediction, BEAM_WIDTH)
            #     for prediction in beam_predictions
            # ]

            # Transform from shape (seq_len, batch_size, feature_dim) to
            # (batch_size, seq_len, feature_dim).
            # beam_predictions = logits.detach().cpu().transpose(0, 1)
            # print('beam_predictions.type', beam_predictions.dtype)

            # print('beam_predictions.shape', beam_predictions.shape)
            # print('target_lengths.shape', target_lengths.shape)
            # print('target_lengths', target_lengths)

            # beam_results, _, _, _ = DECODER.decode(
            #     beam_predictions
            # )
            # print('beam_results.shape', beam_results.shape)

            # # Get the top beam results.
            # top_beam_results = []
            # for i in range(len(beam_results)):
            #     target_length = target_lengths[i]
            #     top_beam_result = beam_results[i, 0, :target_length]
            #     top_beam_results.append(top_beam_result)

            #     # print('target_length', target_length)
            #     print('top_beam_result.shape', top_beam_result.shape)
            #     # print('top_beam_result', top_beam_result)
            #     # print()
            #     # print()

            # beam_decodings = [
            #     "".join([INDEX_TO_CHAR[idx.item()] for idx in beam_result])
            #     for beam_result in top_beam_results
            # ]
            # exit()

            # Calculate the CERs and WERs for the current batch.
            for i, (label, greedy_decoding, beam_decoding) in enumerate(
                zip(labels_to_strings, greedy_decodings, ["beam_decoding"])
            ):
                num_samples += 1
                # assert len(label) == len(beam_decoding)

                # Calculate the CERs and WERs for the current sample.
                greedy_cer = cer(label, greedy_decoding)
                greedy_wer = wer(label, greedy_decoding)
                beam_cer = cer(label, beam_decoding)
                beam_wer = wer(label, beam_decoding)

                # Accumulate the CERs and WERs.
                greedy_total_cer += greedy_cer
                greedy_total_wer += greedy_wer
                beam_total_cer += beam_cer
                beam_total_wer += beam_wer

                # Print the first sample in the batch.
                if i == 0:
                    print("-" * 50)
                    print(f"Label: {label}")
                    print(f"Greedy Decoding Prediction: {greedy_decoding}")
                    print(f"Greedy Decoding CER: {greedy_cer:.4f}")
                    print(f"Greedy Decoding WER: {greedy_wer:.4f}")
                    print(f"Beam Decoding Prediction: {beam_decoding}")
                    print(f"Beam Decoding CER: {beam_cer:.4f}")
                    print(f"Beam Decoding WER: {beam_wer:.4f}")
                    print("-" * 50)
                    print()
                    print()

    # Return the average validation loss, CERs, and WERs.
    return (
        val_loss / num_samples,
        greedy_total_cer / num_samples,
        greedy_total_wer / num_samples,
        beam_total_cer / num_samples,
        beam_total_wer / num_samples,
    )


def _test_model(
    *,
    model: nn.Module,
    criterion: nn.CTCLoss,
    test_loader: DataLoader,
) -> tuple[float, float, float, float, float]:
    """Tests the model on the given test dataset.

    Args:
        model (nn.Module): The model to test.
        criterion (nn.CTCLoss): The CTC loss function.
        test_loader (DataLoader): The test data loader.

    Returns:
        tuple[float, float, float, float, float]:
            The average test loss, greedy decoding CER, greedy decoding WER,
            beam decoding CER, and beam decoding WER.
    """
    # Set the model to evaluation mode.
    model.eval()

    test_loss = 0.0
    num_samples = 0

    greedy_total_cer = 0.0
    greedy_total_wer = 0.0
    beam_total_cer = 0.0
    beam_total_wer = 0.0

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

            # Create an input_lengths tensor for the CTC loss function.
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

            # Calculate the CTC loss and accumulate the test loss.
            loss = criterion(logits, labels, input_lengths, target_lengths)
            test_loss += loss.item()

            # Convert the labels to strings for the CER and WER calculations.
            labels_to_strings = [
                "".join(
                    [
                        INDEX_TO_CHAR[index]
                        for index in label.detach().cpu().numpy()
                    ]
                )
                for label in labels_no_padding
            ]

            # Get the greedy search decodings for the current batch.
            greedy_predictions = logits.argmax(2).detach().cpu().numpy().T
            greedy_decodings = [
                greedy_decode(prediction) for prediction in greedy_predictions
            ]

            # Get the beam search decodings for the current batch.
            # beam_predictions = logits.detach().cpu().numpy()
            # beam_decodings = [
            #     DECODER.decode(prediction, BEAM_WIDTH)
            #     for prediction in beam_predictions
            # ]

            # # Transform from shape (seq_len, batch_size, feature_dim) to
            # # (batch_size, seq_len, feature_dim).
            # beam_predictions = logits.detach().cpu().transpose(0, 1)
            # beam_results, _, _, _ = DECODER.decode(beam_predictions, target_lengths)

            # # Get the top beam results.
            # top_beam_results = []
            # for i in range(len(beam_results)):
            #     target_length = target_lengths[i]
            #     top_beam_result = beam_results[i, 0, :target_length]
            #     top_beam_results.append(top_beam_result)

            # beam_decodings = [
            #     "".join([INDEX_TO_CHAR[idx.item()] for idx in beam_result])
            #     for beam_result in top_beam_results
            # ]

            for i, (label, greedy_decoding, beam_decoding) in enumerate(
                zip(labels_to_strings, greedy_decodings, ["beam_decoding"])
            ):
                num_samples += 1
                # assert len(beam_decoding) == len(label)

                # Calculate the CERs and WERs for the current sample.
                greedy_cer = cer(label, greedy_decoding)
                greedy_wer = wer(label, greedy_decoding)
                beam_cer = cer(label, beam_decoding)
                beam_wer = wer(label, beam_decoding)

                # Accumulate the CERs and WERs.
                greedy_total_cer += greedy_cer
                greedy_total_wer += greedy_wer
                beam_total_cer += beam_cer
                beam_total_wer += beam_wer

                # Print the first sample in the batch.
                if i == 0:
                    print("-" * 50)
                    print(f"Label: {label}")
                    print(f"Greedy Decoding Prediction: {greedy_decoding}")
                    print(f"Greedy Decoding CER: {greedy_cer:.4f}")
                    print(f"Greedy Decoding WER: {greedy_wer:.4f}")
                    print(f"Beam Decoding Prediction: {beam_decoding}")
                    print(f"Beam Decoding CER: {beam_cer:.4f}")
                    print(f"Beam Decoding WER: {beam_wer:.4f}")
                    print("-" * 50)
                    print()
                    print()

    # Return the average test loss, CERs, and WERs.
    return (
        test_loss / num_samples,
        greedy_total_cer / num_samples,
        greedy_total_wer / num_samples,
        beam_total_cer / num_samples,
        beam_total_wer / num_samples,
    )


def train_model(
    model_type: ModelType,
    bezier_curve_degree: int,
    cross_validation: bool,
    num_layers: int,
    bidirectional: bool,
) -> nn.Module:
    """Train and evaluate the given model on the training, validation and test datasets.

    Args:
        model_type (ModelType): The type of model to train.
        bezier_curve_degree (int): The degree of the bezier curves.
        cross_validation (bool, optional): Whether to perform cross validation or not.
        num_layers (int): The number of recurrent layers to use.
        bidirectional (bool): Whether to use bidirectional RNNs or not.

    Returns:
        nn.Module: The trained model.
    """
    wandb.login()

    # Initialize the W&B run.
    wandb.init(
        # Set the project where this run will be logged.
        project="drawing-gui",
        # Track hyperparameters and run metadata.
        config={
            "model_type": model_type.name,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "bezier_curve_degree": bezier_curve_degree,
            "cross_validation": cross_validation,
            "reduction": REDUCTION,
            "num_layers": num_layers,
            "dropout_rate": DROPOUT_RATE,
            "bidirectional": bidirectional,
            "patience": PATIENCE,
            "beam_width": BEAM_WIDTH,
            "device": DEVICE,
        },
    )
    (
        train_cross_val_dataset,
        train_single_val_dataset,
        val_1_dataset,
        val_2_dataset,
        test_dataset,
    ) = _extract_load_datasets(
        bezier_degree=bezier_curve_degree,
        with_cross_val=cross_validation,
    )

    print(
        f"Train Cross Val Dataset Size: {len(train_cross_val_dataset)}\n"
        f"Train Single Val Dataset Size: {len(train_single_val_dataset)}\n"
        f"Val 1 Dataset Size: {len(val_1_dataset)}\n"
        f"Val 2 Dataset Size: {len(val_2_dataset)}\n"
        f"Test Dataset Size: {len(test_dataset)}\n"
    )

    (
        train_cross_val_loader,
        train_single_val_loader,
        val_1_loader,
        val_2_loader,
        test_loader,
    ) = _create_data_loaders(
        train_cross_val_dataset=train_cross_val_dataset,
        train_single_val_dataset=train_single_val_dataset,
        val_1_dataset=val_1_dataset,
        val_2_dataset=val_2_dataset,
        test_dataset=test_dataset,
    )

    # Create the model.
    bezier_curve_dimension = train_cross_val_dataset.all_bezier_curves[
        0
    ].shape[-1]
    model = model_type.value(
        bezier_curve_dimension=bezier_curve_dimension,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
        dropout=DROPOUT_RATE,
        bidirectional=bidirectional,
        device=DEVICE,
    ).to(DEVICE)

    wandb.watch(model)

    # Set up the early stopping callback.
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Set up the loss function and optimizer.
    criterion = nn.CTCLoss(blank=0, reduction=REDUCTION, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the training loop.
    for epoch in range(NUM_EPOCHS):
        # Train the model for one epoch.
        if cross_validation:
            train_loss = _train_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_cross_val_loader,
            )
        else:
            train_loss = _train_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_single_val_loader,
            )

        wandb.log({"Training - Loss": train_loss}, step=epoch)

        # Validate the model on the validation datasets.
        (
            val_1_loss,
            val_1_greedy_cer,
            val_1_greedy_wer,
            val_1_beam_cer,
            val_1_beam_wer,
        ) = _validate_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_1_loader,
        )

        if cross_validation:
            (
                val_2_loss,
                val_2_greedy_cer,
                val_2_greedy_wer,
                val_2_beam_cer,
                val_2_beam_wer,
            ) = _validate_epoch(
                model=model,
                criterion=criterion,
                val_loader=val_2_loader,
            )

            # Calculate the average validation loss, CER and WER.
            avg_epoch_val_loss = (val_1_loss + val_2_loss) / 2
            avg_epoch_val_greedy_cer = (
                val_1_greedy_cer + val_2_greedy_cer
            ) / 2
            avg_epoch_val_greedy_wer = (
                val_1_greedy_wer + val_2_greedy_wer
            ) / 2
            avg_epoch_val_beam_cer = (val_1_beam_cer + val_2_beam_cer) / 2
            avg_epoch_val_beam_wer = (val_1_beam_wer + val_2_beam_wer) / 2
        else:
            # Calculate the average validation loss, CER and WER.
            avg_epoch_val_loss = val_1_loss
            avg_epoch_val_greedy_cer = val_1_greedy_cer
            avg_epoch_val_greedy_wer = val_1_greedy_wer
            avg_epoch_val_beam_cer = val_1_beam_cer
            avg_epoch_val_beam_wer = val_1_beam_wer

        # Log the average validation loss, CERs, and WERs to Weights & Biases.
        wandb.log(
            {
                "Validation - Loss - Per Epoch Average": avg_epoch_val_loss,
                "Validation - Greedy Decoding CER - Per Epoch Average": avg_epoch_val_greedy_cer,
                "Validation - Greedy Decoding WER - Per Epoch Average": avg_epoch_val_greedy_wer,
                "Validation - Beam Decoding CER - Per Epoch Average": avg_epoch_val_beam_cer,
                "Validation - Beam Decoding WER - Per Epoch Average": avg_epoch_val_beam_wer,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {avg_epoch_val_loss:.4f}, "
            f"Val Greedy CER: {avg_epoch_val_greedy_cer:.4f}, "
            f"Val Greedy WER: {avg_epoch_val_greedy_wer:.4f}, "
            f"Val Beam CER: {avg_epoch_val_beam_cer:.4f}, "
            f"Val Beam WER: {avg_epoch_val_beam_wer:.4f}"
        )

        if early_stopping(model, avg_epoch_val_loss):
            print("Early stopping...")
            break

    # Training is complete, evaluate the model on the test dataset.
    (
        test_avg_loss,
        test_avg_greedy_cer,
        test_avg_greedy_wer,
        test_avg_beam_cer,
        test_avg_beam_wer,
    ) = _test_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
    )

    wandb.log(
        {
            "Test - Average Loss": test_avg_loss,
            "Test - Average Greedy CER": test_avg_greedy_cer,
            "Test - Average Greedy WER": test_avg_greedy_wer,
            "Test - Average Beam CER": test_avg_beam_cer,
            "Test - Average Beam WER": test_avg_beam_wer,
        }
    )

    print(
        f"Test Average Loss: {test_avg_loss:.4f}, "
        f"Test Average Greedy CER: {test_avg_greedy_cer:.4f}, "
        f"Test Average Greedy WER: {test_avg_greedy_wer:.4f}, "
        f"Test Average Beam CER: {test_avg_beam_cer:.4f}, "
        f"Test Average Beam WER: {test_avg_beam_wer:.4f}"
    )

    # Create a models directory if it doesn't exist.
    Path("backend/iam/models").mkdir(parents=True, exist_ok=True)

    trained_model_file_name = get_model_file_name(
        model_name=model_type.name.lower(),
        bezier_curve_degree=bezier_curve_degree,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    # Save the trained model.
    torch.save(
        model.state_dict(),
        f"backend/iam/models/{trained_model_file_name}.ckpt",
    )

    return model


if __name__ == "__main__":
    train_model(
        model_type=ModelType.LSTM,
        bezier_curve_degree=BEZIER_CURVE_DEGREE,
        cross_validation=CROSS_VALIDATION,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
    )
