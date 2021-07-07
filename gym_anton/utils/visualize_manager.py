from torch.utils.tensorboard import SummaryWriter


class TensorboardManager:
    def __init__(self):
        self.writer = SummaryWriter()

    def add(self, iter, info):
        for k, v in info.items():
            self.writer.add_scalar(k, v, iter)
