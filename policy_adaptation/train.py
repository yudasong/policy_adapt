import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def update_model(SAs, SPs, model, agnostic_size, num_epoch=5, decay=0, is_policy=False, mini_batch_size=64):

    error_criteria = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005 - 0.005 * (decay), eps=1e-5)

    losses = []

    sampler = BatchSampler(
        SubsetRandomSampler(range(SAs.shape[0])),
        mini_batch_size,
        drop_last=True)

    for i in range(num_epoch):

        for index in sampler:

            x = torch.tensor(SAs[index])
            y = torch.tensor(SPs[index])
            x = torch.squeeze(x)
            y = torch.squeeze(y)
            if is_policy:
                recurrent_hidden_states = torch.zeros(x.shape[0],model.recurrent_hidden_state_size)
                masks = torch.zeros(x.shape[0], 1)
                _, log_prob, _, _ = model.evaluate_actions(x[:,agnostic_size:], recurrent_hidden_states, masks, y)
                loss = -log_prob.mean()
            else:
                loss = error_criteria(model.predict(x),y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return np.mean(np.asarray(losses))
