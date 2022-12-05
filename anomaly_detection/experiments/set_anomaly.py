import mate
from ..models.transformer import Transformer
from ..data_loaders.set_anomaly import get_loaders
from ..trainers.anomaly_trainer import AnomalyTrainer
from jax import random

random_key = random.PRNGKey(0)
train_loader, val_loader, test_loader, _ = get_loaders("data", random_key)

model = Transformer(
    model_dim=256,
    num_heads=4,
    num_layers=1,
    num_classes=1,
    dropout_prob=0.1,
    input_dropout_prob=0.1,
)

if mate.is_train:
    val_acc = AnomalyTrainer.train(
        model=model,
        max_epochs=10,
        anom_train_loader=train_loader,
        anom_val_loader=val_loader,
        seed=0,
        checkpoint_path=mate.default_checkpoint_location,
    )
