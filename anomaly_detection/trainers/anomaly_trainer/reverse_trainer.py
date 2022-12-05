import jax
from jax import random
import optax
from .trainer import TrainerModule
from flax import linen as nn


class ReverseTrainer(TrainerModule):
    def batch_to_input(self, batch):
        inp_data, _ = batch
        inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
        return inp_data

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            inp_data, labels = batch
            inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply(
                {"params": params},
                inp_data,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, rng)

        return calculate_loss

    @staticmethod
    def train_reverse(
        *,
        checkpoint_path: str,
        seed: int,
        model: nn.Module,
        rev_train_loader,
        rev_val_loader,
        max_epochs=10,
        lr=1e-3,
        warmup=50
    ):
        num_train_iters = len(rev_train_loader) * max_epochs
        # Create a trainer module with specified hyperparameters
        trainer = ReverseTrainer(
            lr=lr,
            warmup=warmup,
            model=model,
            seed=seed,
            checkpoint_path=checkpoint_path,
            exmp_batch=next(iter(rev_train_loader)),
            max_iters=num_train_iters,
        )
        if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
            trainer.train_model(rev_train_loader, rev_val_loader, num_epochs=max_epochs)
            trainer.load_model()
        else:
            trainer.load_model(pretrained=True)
        val_acc = trainer.eval_model(rev_val_loader)
        # Bind parameters to model for easier inference
        trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
        return val_acc

    @staticmethod
    def test_reverse(model, seed, checkpoint_path, rev_test_loader, max_epochs=10):
        num_train_iters = len(rev_test_loader) * max_epochs
        # Create a trainer module with specified hyperparameters
        trainer = ReverseTrainer(
            model=model,
            seed=seed,
            checkpoint_path=checkpoint_path,
            exmp_batch=next(iter(rev_test_loader)),
            max_iters=num_train_iters,
        )
        assert trainer.checkpoint_exists()
        trainer.load_model(pretrained=True)
        test_acc = trainer.eval_model(rev_test_loader)
        # Bind parameters to model for easier inference
        trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
        return test_acc
