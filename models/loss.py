import torch
import models
import torch.nn as nn
import data.dict as dict


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight)
    if use_cuda:
        crit.cuda()
    return crit


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = torch.tensor(hidden_outputs.data, requires_grad=True)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        targ_t = targ_t.view(-1)
        loss_t = criterion(scores_t, targ_t)
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.item()
        loss_t.div(num_total_t.float()).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss


def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = decoder.compute_score(outputs)
    targets = targets.view(-1)
    loss = criterion(scores, targets) + sim_score
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss.div(num_total.float()).backward()
    loss = loss.item()

    return loss
