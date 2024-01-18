from torch import nn, optim
from torch.optim import Adam

from models.model.JigsawNet import JIGSAW_NET
from models.utils.conf import *
from models.datasets.dataset import DATA_LOADER

model = JIGSAW_NET(3)
TRUE_LABEL = list(range(16))

optimizer = Adam(params=model.parameters(), lr = INIT_LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True,
                                                 factor=FACTOR, patience=PATIENCE)

criterion = nn.CrossEntropyLoss()

def train(model, datasets, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(datasets):
        output = model(batch)