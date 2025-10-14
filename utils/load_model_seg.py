import torch
import logging
from model.tinyusfm_seg import SegmentationModel as TinyUSFM_Seg
from model.usfm_seg import VisionTransformer as USFM_seg


logger = logging.getLogger(__name__)

def load_model_seg(args, device):
    pretrained = args.pretrained == 'True'
    if args.model_name == "TinyUSFM_Seg":
        model = TinyUSFM_Seg(args.num_classes)  
        if pretrained and hasattr(args, 'checkpoint') and args.checkpoint:
            checkpoint_path = args.checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            new_state_dict = {
                k.replace('model.', 'backbone.'): v
                for k, v in checkpoint.items()
                if k.startswith('model.') 
            }
            load_info = model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            if load_info.missing_keys:
                logger.info(f"Missing keys (not initialized from checkpoint): {load_info.missing_keys}")
            if load_info.unexpected_keys:
                logger.info(f"Unexpected keys (checkpoint has but model does not need): {load_info.unexpected_keys}")
    elif args.model_name == "USFM_Seg":
        model = USFM_seg(num_classes=args.num_classes)
        if pretrained and hasattr(args, 'checkpoint') and args.checkpoint:
            checkpoint_path = args.checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint
            new_state_dict = {
                f'transformer.{k}': v
                for k, v in state_dict.items()
            }
            load_info = model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            if load_info.missing_keys:
                logger.info(f"Missing keys (not initialized from checkpoint): {load_info.missing_keys}")
            if load_info.unexpected_keys:
                logger.info(f"Unexpected keys (checkpoint has but model does not need): {load_info.unexpected_keys}")
    else:
        raise ValueError(f"Unsupported model_name '{args.model_name}'. " "Please use 'USFM' or 'TinyUSFM'.")
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {args.model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    return model