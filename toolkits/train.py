from dataset import churn_dataset
from config import config
from tqdm import trange, tqdm
from net import Net
import mindspore as ms
from sklearn.metrics import roc_auc_score
from mindspore import nn, ops, value_and_grad
from mindspore.experimental import optim

train_set = churn_dataset(config["train_set"], "train").batch(config["batch_size"])
dev_set = churn_dataset(config["dev_set"], "dev").batch(config["batch_size"])

model = Net()
loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(model.trainable_params(), lr=config["lr"])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=2, min_lr=1e-3
)


def forward_fn(x, y):
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss


backward_fn = value_and_grad(forward_fn, None, model.trainable_params())


def train_step(x, y):
    loss, grads = backward_fn(x, y)
    optimizer(grads)
    return loss


best = 0
for epoch in trange(config["epoches"]):
    train_loss = 0
    for x, y in train_set.create_tuple_iterator():
        train_loss += train_step(x, y)

    dev_loss = 0
    preds = ms.tensor([], ms.int32)
    labels = ms.tensor([], ms.int32)
    

    for x, y in dev_set.create_tuple_iterator():
        logits = model(x)
        y_hat = logits.argmax(axis=1)

        preds = ops.concat((preds, y_hat.astype(preds.dtype)), axis=0)
        labels = ops.concat((labels, y.astype(labels.dtype)), axis=0)
        score = roc_auc_score(labels.asnumpy(), preds.asnumpy())
        dev_loss += loss_fn(model(x), y)

    lr_scheduler.step(dev_loss)
    if score >= best:
        best = score
        ms.save_checkpoint(model, config["model_path"] + "best.ckpt")
        tqdm.write(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Dev Loss: {dev_loss}, AUC: {score},lr: {lr_scheduler.get_last_lr()[0]}, Saved!"
        )
    else:
        tqdm.write(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Dev Loss: {dev_loss}, AUC: {score},lr: {lr_scheduler.get_last_lr()[0]}"
        )
