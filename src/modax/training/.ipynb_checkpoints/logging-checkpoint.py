from tensorboardX import SummaryWriter
#from tensorflow.summary import create_file_writer
#from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

    def write(self, metrics, epoch):
        for key, value in metrics.items():
            if value.squeeze().ndim == 0:
                self.writer.add_scalar(key, value, epoch)
            elif value.squeeze().ndim == 1:
                components = {
                    f"comp_{idx}": comp for idx, comp in enumerate(value.squeeze())
                }
                self.writer.add_scalars(
                    key, components, epoch,
                )
            else:
                raise NotImplementedError

    def close(self,):
        self.writer.flush()
        self.writer.close()
