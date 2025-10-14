import torch


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        end_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs        
            return [self.base_lr * factor for _ in self.optimizer.param_groups]

        poly_epoch = epoch - self.warmup_epochs
        poly_total = self.max_epochs - self.warmup_epochs
        factor = (1 - poly_epoch / poly_total) ** self.power  
        return [
            self.end_lr + (self.base_lr - self.end_lr) * factor
            for _ in self.optimizer.param_groups
        ]


def build_scheduler(optimizer, args):
    return WarmupPolyLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.num_epochs,
        base_lr=args.lr,
        end_lr=1e-6, 
        power=1.0   
    )
