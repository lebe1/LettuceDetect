import time
from datetime import timedelta

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from lettucedetect.models.evaluator import evaluate_model, print_metrics


class Trainer:
    """Token classification trainer with epoch-based training and validation.

    Trains a model using AdamW, evaluates on a test set after each epoch,
    and saves the best checkpoint based on hallucinated-class F1.
    """

    def __init__(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 6,
        learning_rate: float = 1e-5,
        save_path: str = "best_model",
        device: torch.device | None = None,
    ):
        """Initialize the trainer.

        :param model: The model to train
        :param tokenizer: Tokenizer for the model
        :param train_loader: DataLoader for training data
        :param test_loader: DataLoader for test data
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for optimization
        :param save_path: Path to save the best model
        :param device: Device to train on (defaults to cuda if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        self.optimizer: Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.to(self.device)

    def train(self) -> float:
        """Train the model.

        Returns:
            Best F1 score achieved during training

        """
        best_f1: float = 0
        start_time = time.time()

        print(f"\nStarting training on {self.device}")
        print(
            f"Training samples: {len(self.train_loader.dataset)}, "
            f"Test samples: {len(self.test_loader.dataset)}\n"
        )

        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            self.model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc="Training", leave=True)

            for batch in progress_bar:
                self.optimizer.zero_grad()
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / num_batches:.4f}",
                    }
                )

            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1} completed in {timedelta(seconds=int(epoch_time))}. Average loss: {avg_loss:.4f}"
            )

            print("\nEvaluating...")
            metrics = evaluate_model(self.model, self.test_loader, self.device)
            print_metrics(metrics)

            if metrics["hallucinated"]["f1"] > best_f1:
                best_f1 = metrics["hallucinated"]["f1"]
                self.model.save_pretrained(self.save_path)
                self.tokenizer.save_pretrained(self.save_path)
                print(f"\n🎯 New best F1: {best_f1:.4f}, model saved at '{self.save_path}'!")

            print("-" * 50)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
        print(f"Best F1 score: {best_f1:.4f}")

        return best_f1
