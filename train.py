"""
COVID-19 Detection from Cough Audio — Main Training Script
===========================================================
Trains an ECAPA-TDNN model to classify cough recordings as
COVID-19 Positive or Negative using SpeechBrain.

Usage
-----
    python train.py train.yaml

Author: Sami Bahig, MD MSc — Université de Montréal / MILA
"""

import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from preparation.prepare_data import prepare_covid_dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Brain Class — Training & Evaluation Logic
# ─────────────────────────────────────────────

class CovidBrain(sb.Brain):
    """SpeechBrain Brain class for COVID-19 cough classification.

    Handles the full training loop:
        1. Feature extraction (log-Mel filterbanks)
        2. Normalization
        3. ECAPA-TDNN embedding
        4. Classification (COVID+ / COVID-)
        5. Loss computation & optimization
    """

    def compute_forward(self, batch, stage):
        """Forward pass — extract features and compute class logits.

        Arguments
        ---------
        batch : PaddedBatch
            Batch of audio signals and metadata.
        stage : sb.Stage
            Current stage (TRAIN, VALID, TEST).

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch, 1, n_classes).
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # 1. Extract log-Mel filterbank features
        feats = self.modules.compute_features(wavs)

        # 2. Normalize features
        feats = self.modules.mean_var_norm(feats, lens)

        # 3. Compute ECAPA-TDNN embeddings
        embeddings = self.modules.embedding_model(feats, lens)

        # 4. Classify
        logits = self.modules.classifier(embeddings)

        return logits

    def compute_objectives(self, predictions, batch, stage):
        """Compute NLL loss and track classification error.

        Arguments
        ---------
        predictions : torch.Tensor
            Class logits from compute_forward.
        batch : PaddedBatch
            Current batch.
        stage : sb.Stage
            Current stage.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        _, lens = batch.sig
        status = batch.status_encoded.squeeze(1)

        loss = self.hparams.compute_cost(
            predictions, status, length=lens
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                batch.id, predictions, status, lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Initialize metrics at the start of each stage."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Log results and save checkpoints at end of each stage."""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        elif stage == sb.Stage.VALID:
            stage_stats["error"] = self.error_metrics.summarize("average")

            # Learning rate scheduling
            old_lr, new_lr = self.hparams.lr_scheduler(
                [self.optimizer], epoch, stage_loss
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Log to file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"error": stage_stats["error"]},
                min_keys=["error"],
            )

        elif stage == sb.Stage.TEST:
            stage_stats["error"] = self.error_metrics.summarize("average")
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                test_stats=stage_stats,
            )
            logger.info(
                f"Test Error: {stage_stats['error']:.4f} | "
                f"Test Loss: {stage_loss:.4f}"
            )


# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────

def data_io_prep(hparams):
    """Prepares SpeechBrain datasets from JSON manifests.

    Arguments
    ---------
    hparams : dict
        Hyperparameters loaded from train.yaml.

    Returns
    -------
    tuple : (train_data, valid_data, test_data)
    """
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define label pipeline
    @sb.utils.data_pipeline.takes("status")
    @sb.utils.data_pipeline.provides("status_encoded")
    def label_pipeline(status):
        yield torch.LongTensor([status])

    # Load datasets
    datasets = {}
    for split, json_path in [
        ("train", hparams["train_annotation"]),
        ("valid", hparams["valid_annotation"]),
        ("test",  hparams["test_annotation"]),
    ]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "status_encoded"],
        )

    return datasets["train"], datasets["valid"], datasets["test"]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load hyperparameters from YAML
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    # Create output directories
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare dataset manifests
    run_on_main(
        prepare_covid_dataset,
        kwargs={
            "data_folder":     hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test":  hparams["test_annotation"],
            "split_ratio":     hparams.get("split_ratio", [80, 10, 10]),
        },
    )

    # Load datasets
    train_data, valid_data, test_data = data_io_prep(hparams)

    # Sort training data by duration for efficient batching
    train_data = train_data.filtered_sorted(sort_key="length")

    # Initialize Brain
    covid_brain = CovidBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    covid_brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={"batch_size": hparams["batch_size"]},
        valid_loader_kwargs={"batch_size": hparams["batch_size"]},
    )

    # Evaluate on test set
    covid_brain.evaluate(
        test_set=test_data,
        min_key="error",
        test_loader_kwargs={"batch_size": hparams["batch_size"]},
    )
