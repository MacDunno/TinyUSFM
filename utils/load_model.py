import torch  
from model.usfm import VisionTransformer  
from functools import partial  
import torch.nn as nn  
import logging  
from model.tinyusfm import TinyUSFM


logger = logging.getLogger(__name__)  

def load_model(args, device):  
    pretrained = args.pretrained == 'True'  
    if args.model_name == "USFM":  
        model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6))  
        checkpoint_path = args.checkpoint 
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  
        load_info = model.load_state_dict(checkpoint, strict=False)  
        if load_info.missing_keys:
            logger.info(f"Missing keys (not initialized from checkpoint): {load_info.missing_keys}")
        if load_info.unexpected_keys:
            logger.info(f"Unexpected keys (checkpoint has but model does not need): {load_info.unexpected_keys}")

    elif args.model_name == "TinyUSFM":
        model = TinyUSFM(num_classes=args.num_classes, global_pool=True)
        if pretrained:
            checkpoint_path = args.checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            fixed_weights = {}
            for key, value in checkpoint.items():
                if key.startswith('module.model.'):
                    new_key = 'model.' + key[13:]
                elif key.startswith('module.'):
                    new_key = 'model.' + key[7:]
                else:
                    new_key = key
                fixed_weights[new_key] = value
            fixed_weights = {k: v for k, v in fixed_weights.items()
                 if not (k.endswith('head.weight') or k.endswith('head.bias'))}
            load_info = model.load_state_dict(fixed_weights, strict=False)
            if load_info.missing_keys:
                logger.info(f"Missing keys (not initialized from checkpoint): {load_info.missing_keys}")
            if load_info.unexpected_keys:
                logger.info(f"Unexpected keys (checkpoint has but model does not need): {load_info.unexpected_keys}")
    else:  
        raise ValueError(f"Unsupported model_name '{args.model_name}'. " "Please use 'USFM' or 'TinyUSFM'.")
    model = model.to(device)  
    return model
