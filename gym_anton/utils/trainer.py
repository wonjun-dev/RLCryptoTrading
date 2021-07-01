import torch.nn.functional as F


def dqn_trainer(q, q_target, memory, optimizer, gamma, iteration=30, batch_size=128):
    for i in range(iteration):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
