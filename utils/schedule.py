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

def get_lr_decay_param_groups(model, base_lr, weight_decay, num_layers=12, layer_decay=0.8):
    def get_layer_id(param_name):
        if param_name.startswith("backbone"):
            if "blocks." in param_name:
                block_id = int(param_name.split("blocks.")[1].split(".")[0])
                return block_id
            elif "patch_embed" in param_name:
                return 0
            else:
                return num_layers - 1
        else:
            return num_layers 

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        group_name = f"layer_{layer_id}"

        if group_name not in param_groups:
            scale = layer_decay ** (num_layers - layer_id)
            param_groups[group_name] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay
            }

        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())