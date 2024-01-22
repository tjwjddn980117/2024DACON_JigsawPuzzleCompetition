import time

from torch import nn, optim
from torch.optim import Adam

from models.model.JigsawNet import JIGSAW_NET
from models.utils.conf import *
from models.datasets.dataset import TRAIN_DATA_LOADER, TEST_DATA_LOADER

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

        optimizer.zero_grad()
        output = model(batch)
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        loss = criterion(output, output_reshape)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(datasets)) * 100, 2), '% , loss :', loss.item())
    
    return epoch_loss / len(datasets)

def evaluation(model, datasets, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(datasets):
            output = model(batch)
            output_reshape = output.contiguous().view(-1, output.shape[-1])

            loss = criterion(output, output_reshape)
            loss.backward()

            epoch_loss += loss.item()
        
    return epoch_loss / len(datasets)

def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, TRAIN_DATA_LOADER, optimizer, criterion)
        valid_loss = evaluation(model, TEST_DATA_LOADER, criterion)
        end_time = time.time()