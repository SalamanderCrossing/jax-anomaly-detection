import optax
from jax import random
import jax.numpy as jnp
from .trainer import TrainerModule
from flax import linen as nn
from torch.utils.data import DataLoader


class AnomalyTrainer(TrainerModule):
    def batch_to_input(self, batch):
        inp_data, _, _ = batch
        return inp_data

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            inp_data, _, labels = batch
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply(
                {"params": params},
                inp_data,
                add_positional_encoding=False,  # No positional encoding since this is a permutation equivariant task
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            logits = logits.squeeze(axis=-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            acc = (logits.argmax(axis=-1) == labels).astype(jnp.float32).mean()
            return loss, (acc, rng)

        return calculate_loss

    @staticmethod
    def train(
        *,
        model: nn.Module,
        max_epochs=100,
        anom_train_loader: DataLoader,
        anom_val_loader: DataLoader,
        checkpoint_path: str,
        seed: int
    ):
        num_train_iters = len(anom_train_loader) * max_epochs
        # Create a trainer module with specified hyperparameters
        trainer = AnomalyTrainer(
            exmp_batch=next(iter(anom_train_loader)),
            max_iters=num_train_iters,
            model=model,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
            trainer.train_model(
                anom_train_loader, anom_val_loader, num_epochs=max_epochs
            )
            trainer.load_model()
        else:
            trainer.load_model(pretrained=True)
        # train_acc = trainer.eval_model(anom_train_loader)
        val_acc = trainer.eval_model(anom_val_loader)
        # Bind parameters to model for easier inference
        trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
        # return trainer, {
        #     "train_acc": train_acc,
        #     "val_acc": val_acc,
        #     "test_acc": test_acc,
        # }
        return val_acc

    @staticmethod
    def test(
        model: nn.Module, anom_test_loader: DataLoader, checkpoint_path: str, seed: int
    ):
        # Create a trainer module with specified hyperparameters
        trainer = AnomalyTrainer(
            exmp_batch=next(iter(anom_test_loader)),
            model=model,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        assert trainer.checkpoint_exists()
        trainer.load_model(pretrained=True)
        test_acc = trainer.eval_model(anom_test_loader)
        # Bind parameters to model for easier inference
        trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
        test_acc = trainer.eval_model(anom_test_loader)
        # return trainer, {
        #     "train_acc": train_acc,
        #     "val_acc": val_acc,
        #     "test_acc": test_acc,
        # }
        return test_acc
