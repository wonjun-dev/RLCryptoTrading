import collections
import random

import torch


class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float, device=self.device),
            torch.tensor(a_lst, device=self.device),
            torch.tensor(r_lst, device=self.device),
            torch.tensor(s_prime_lst, dtype=torch.float, device=self.device),
            torch.tensor(done_mask_lst, device=self.device),
        )

    def size(self):
        return len(self.buffer)


if __name__ == "__main__":
    memory = ReplayBuffer(10)
    s, a, r, s_prime, done_mask = 1, 0, 10, 2, 0
    memory.put((s, a, r, s_prime, done_mask))
    print(memory.size())
    transition = memory.sample(1)
    print(transition)
    print(memory.size())
