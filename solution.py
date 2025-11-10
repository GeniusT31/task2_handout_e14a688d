import abc
import collections
import enum
import math
import pathlib
import typing
import warnings
import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt
from util import paint_reliability_diagram, compute_cost, seed_setup, calculate_calibration_curve

ENABLE_EXTENDED_ANALYSIS = False
"""
Set ENABLE_EXTENDED_ANALYSIS to True in order to generate additional plots on validation data.
"""

USE_PRETRAINED_WEIGHTS = True
"""
If USE_PRETRAINED_WEIGHTS is True, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the baseline.
If you set the constant to False (to further experiment), this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():
    raise RuntimeError(
        "This main() method is for illustrative purposes only"
        " and will NEVER be called when running your solution to generate your submission file!\n"
        "The checker always directly interacts with your SWAInferenceHandler class and run_evaluation method.\n"
        "You can remove this exception for local testing, but be aware that any changes to the main() method"
        " are ignored when generating your submission file."
    )

    data_location = pathlib.Path.cwd()
    model_location = pathlib.Path.cwd()
    output_location = pathlib.Path.cwd()

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    seed_setup()

    # Build and run the actual solution
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    swag_inference = SWAInferenceHandler(
        train_xs=training_dataset.tensors[0],
        model_dir=model_location,
    )

    swag_inference.train_model(training_loader)
    swag_inference.run_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    with torch.random.fork_rng():
        run_evaluation(swag_inference, validation_dataset, ENABLE_EXTENDED_ANALYSIS, output_location)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    MAP simply predicts the most likely class using pretrained MAP weights.
    SWAG_DIAGONAL and SWAG_FULL correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAInferenceHandler(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        # TODO(2): optionally add/tweak hyperparameters
        swag_training_epochs: int = 30,
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 30,
    ):
        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples

        # Network used to perform SWAG.
        self.network = CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal - TODO(1): create attributes for SWAG-diagonal
        if inference_mode in (InferenceMode.SWAG_DIAGONAL, InferenceMode.SWAG_FULL):
            self.swag_mean = self._create_weight_copy()
            self.swag_mean_sq = self._create_weight_copy()
            self.swag_n = 0  # number of collected snapshots

        # Full SWAG - TODO(2): create attributes for SWAG-full
        if inference_mode == InferenceMode.SWAG_FULL:
            # Use deque to store deviation vectors with max length
            self.swag_deviation_matrix = collections.deque(maxlen=max_rank_deviation_matrix)

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._calibration_threshold = None

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """
        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in copied_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight name using copied_params and param
            # update mean
            self.swag_mean[name] = (self.swag_mean[name] * self.swag_n + param) / (self.swag_n + 1)
            # update mean_sq
            self.swag_mean_sq[name] = (self.swag_mean_sq[name] * self.swag_n + (param * param)) / (self.swag_n + 1)

        # increment
        self.swag_n += 1

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            # TODO(2): update full SWAG attributes for weight name using copied_params and param
            # Compute deviation from mean and store it
            deviation = {}
            for name, param in copied_params.items():
                deviation[name] = param - self.swag_mean[name]
            
            # Append to deque (automatically removes oldest if at max capacity)
            self.swag_deviation_matrix.append(deviation)

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """
        # We use SGD with momentum and weight decay to perform SWA.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )

        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )

        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        # TODO(1): Perform initialization for SWAG fitting
        self.swag_mean = self._create_weight_copy()
        self.swag_mean_sq = self._create_weight_copy()
        self.swag_n = 0
        if self.inference_mode == InferenceMode.SWAG_FULL:
            self.swag_deviation_matrix = collections.deque(maxlen=self.max_rank_deviation_matrix)

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0

                for batch_images, batch_snow_labels, batch_cloud_labels, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                progress_dict["avg. epoch loss"] = avg_loss
                progress_dict["avg. epoch accuracy"] = avg_accuracy
                pbar.set_postfix(progress_dict)

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                if epoch % self.swag_update_interval == 0:
                    self.update_swag_statistics()

    def run_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        """
        if self.inference_mode == InferenceMode.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._calibration_threshold = 0.0
            return

        # TODO(1): pick a prediction threshold, either constant or adaptive.
        self._calibration_threshold = 2.0 / 3.0

        # TODO(2): perform additional calibration if desired.
        val_images, val_snow_labels, val_cloud_labels, val_labels = validation_data.tensors
        assert val_images.size() == (140, 3, 60, 60)  # N x C x H x W
        assert val_labels.size() == (140,)
        assert val_snow_labels.size() == (140,)
        assert val_cloud_labels.size() == (140,)

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        """
        self.network.eval()

        # Perform Bayesian model averaging
        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            # TODO(1): Sample new parameters for self.network from the SWAG approximate posterior
            self.sample_parameters()

            # TODO(1): Perform inference for all samples in loader using current model sample
            model_sample_predictions = []
            with torch.no_grad():  # disable gradient
                for (batch_x,) in loader:
                    pred_y = self.network(batch_x)
                    # get probability
                    pred_p = torch.softmax(pred_y, dim=-1)
                    model_sample_predictions.append(pred_p)

            # concatenate predictions across all batches
            model_sample_predictions = torch.cat(model_sample_predictions)
            # save results
            model_predictions.append(model_sample_predictions)

        assert len(model_predictions) == self.num_bma_samples
        assert all(
            isinstance(sample_predictions, torch.Tensor)
            and sample_predictions.dim() == 2  # N x C
            and sample_predictions.size(1) == 6
            for sample_predictions in model_predictions
        )

        # TODO(1): Average predictions from different model samples into bma_probabilities
        bma_probabilities = torch.stack(model_predictions).mean(dim=0)

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        return bma_probabilities

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        """
        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_diag = torch.randn(param.size())

            # TODO(1): Sample parameter values for SWAG-diagonal
            mean_weights = self.swag_mean[name]
            var_weights = self.swag_mean_sq[name] - (mean_weights ** 2)
            std_weights = torch.sqrt(torch.clamp(var_weights, min=1e-8))  # avoid negative variance

            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part
            sampled_weight = mean_weights + std_weights * z_diag

            # Full SWAG part
            if self.inference_mode == InferenceMode.SWAG_FULL:
                # TODO(2): Sample parameter values for full SWAG
                # Sample from low-rank component
                K = len(self.swag_deviation_matrix)
                if K > 0:
                    # Sample z_lr ~ N(0, I_K)
                    z_lr = torch.randn(K) / math.sqrt(2.0 * (K - 1))
                    
                    # Compute low-rank contribution: (1/sqrt(2(K-1))) * D @ z_lr
                    low_rank_component = torch.zeros_like(param)
                    for i, deviation in enumerate(self.swag_deviation_matrix):
                        low_rank_component += z_lr[i] * deviation[name]
                    
                    sampled_weight += low_rank_component

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_weight

        # TODO(1): Don't forget to update batch normalization statistics
        self._update_batchnorm_statistics()

    def label_prediction(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        """
        # label_probabilities contains the per-row maximum values in predicted_probabilities,
        # max_likelihood_labels the corresponding column index (equivalent to class).
        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1)
        num_samples, num_classes = predicted_probabilities.size()
        assert label_probabilities.size() == (num_samples,) and max_likelihood_labels.size() == (num_samples,)

        # TODO(2): implement a different decision rule if desired
        return torch.where(
            label_probabilities >= self._calibration_threshold,
            max_likelihood_labels,
            torch.ones_like(max_likelihood_labels) * -1,
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def train_model(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        """
        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_WEIGHTS:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceMode.SWAG_DIAGONAL, InferenceMode.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )

        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0

                for batch_images, _, _, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                progress_dict["avg. epoch loss"] = avg_loss
                progress_dict["avg. epoch accuracy"] = avg_accuracy
                pbar.set_postfix(progress_dict)

    def predict_probs(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        """
        self.network = self.network.eval()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():
            if self.inference_mode == InferenceMode.MAP:
                return self.predict_probabilities_map(loader)
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))
        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        """
        original_momentum_values = dict()
        for module in self.network.modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            original_momentum_values[module] = module.momentum
            module.momentum = None
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        """
        # TODO(2): Implement a custom schedule if desired
        return previous_lr

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use get_last_lr().",
                UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def run_evaluation(
    swag_inference: SWAInferenceHandler,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Run evaluation with your model.
    """
    print("Evaluating model on validation data")

    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    all_pred_probabilities = swag_inference.predict_probs(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)
    predicted_labels = swag_inference.label_prediction(all_pred_probabilities)

    non_ambiguous_mask = labels != -1

    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()

    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    threshold_values = [0.0] + list(torch.unique(max_pred_probabilities, sorted=True))
    costs = []
    for threshold in threshold_values:
        thresholded_predictions = torch.where(max_pred_probabilities <= threshold, -1 * torch.ones_like(predicted_labels), predicted_labels)
        costs.append(compute_cost(thresholded_predictions, labels).item())

    best_threshold_index = np.argmin(costs)
    print(f"Best cost {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    calibration_data = calculate_calibration_curve(all_pred_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        print("Plotting reliability diagram")
        fig = paint_reliability_diagram(calibration_data)
        fig.savefig(output_location / "reliability_diagram.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6),
                    all_pred_probabilities[sample_index].numpy(),
                    tick_label=np.arange(6),
                    color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / "examples_most_confident.pdf")

        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6),
                    all_pred_probabilities[sample_index].numpy(),
                    tick_label=np.arange(6),
                    color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)
        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        log_softmax = self.linear(x)
        return log_softmax


if __name__ == "__main__":
    main()