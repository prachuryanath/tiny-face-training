import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F 
from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric
from ..utils.config import configs
from ..utils import dist
from ..utils.logging import logger

# Import verification helpers
try:
    from ..utils.verification import load_bin, test
except ImportError:
    print("Warning: algorithm.verification could not be imported. Validation will fail.")
    print("Please ensure 'verification.py' is in the 'algorithm/' directory.")
    load_bin = None
    test = None


class FaceRecognitionTrainer(BaseTrainer):

    def __init__(self, model, data_loader, criterion, optimizer, lr_scheduler):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)

        self.verification_sets = {}
        if dist.rank() == 0: # Only load validation sets on main process
            self.load_verification_datasets()

    def load_verification_datasets(self):
        # ... (this function is unchanged) ...
        if load_bin is None:
            logger.info("Skipping validation dataset loading as verification.py is not found.")
            return
            
        ver_targets_dir = configs.data_provider.get('verification_targets_dir')
        ver_targets = configs.data_provider.get('verification_targets', [])
        image_size = configs.data_provider.get('eval_image_size', [112, 112])
        
        if not ver_targets_dir:
            logger.info("Warning: 'data_provider.verification_targets_dir' is not set. Skipping validation.")
            return

        logger.info(f"Loading verification datasets from {ver_targets_dir}...")
        for target_name in ver_targets:
            bin_path = os.path.join(ver_targets_dir, f"{target_name}.bin")
            if os.path.exists(bin_path):
                logger.info(f"Loading {target_name}.bin ...")
                data_set = load_bin(bin_path, image_size)
                self.verification_sets[target_name] = data_set
                logger.info(f"Loaded {target_name} dataset.")
            else:
                logger.info(f"Warning: Could not find {bin_path}")
        
        logger.info(f"Loaded {len(self.verification_sets)} verification datasets.")

    def validate(self):
        # ... (this function is unchanged) ...
        # Validation is only performed on the main rank (rank 0)
        if dist.rank() > 0 or not self.verification_sets:
            # Other ranks just return a placeholder
            # This ensures they don't block on checkpoint saving, etc.
            # The 'val/top1' is used for checkpointing.
            return {'val/top1': 0.0} 

        self.model.eval() # Set model to evaluation mode

        val_results = {}
        primary_metric = 0.0 # Used for checkpointing (e.g., LFW accuracy)

        # Get the "module" from DistributedDataParallel if it exists
        model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Use validation batch size from config, default to training bs
        val_batch_size = configs.data_provider.get('val_batch_size', configs.data_provider.base_batch_size)
        if val_batch_size is None: # Fallback if base_batch_size is also None
            val_batch_size = 32 

        for target_name, data_set in self.verification_sets.items():
            logger.info(f"Running verification on: {target_name}")
            try:
                acc1, std1, acc2, std2, xnorm, _ = test(
                    data_set, model_to_eval, val_batch_size, nfolds=10
                )
                
                logger.info(f"[{target_name}] Accuracy-Flip: {acc2:.5f} +/- {std2:.5f}")
                logger.info(f"[{target_name}] XNorm: {xnorm:.5f}")
                
                val_results[f'val/{target_name}_acc'] = acc2
                val_results[f'val/{target_name}_std'] = std2
                
                if target_name == 'lfw': # Use LFW as the primary metric for saving best model
                    primary_metric = acc2

            except Exception as e:
                logger.info(f"Error during validation on {target_name}: {e}")
                val_results[f'val/{target_name}_acc'] = 0.0
                val_results[f'val/{target_name}_std'] = 0.0
        
        # 'val/top1' is the key the BaseTrainer uses to save the best checkpoint
        # We map our primary metric (LFW acc) to this key.
        val_results['val/top1'] = primary_metric
        
        # Put model back in train mode
        self.model.train()

        return val_results

    def train_one_epoch(self, epoch):
        self.model.train()
        
        # --- SAMPLER LOGIC ---
        # Get the correct sampler to set the epoch
        train_loader = self.data_loader['train']
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch) # For PKSampler
        elif hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch) # For DistributedSampler
        # --- END SAMPLER LOGIC ---

        train_loss = DistributedMetric('train_loss')

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            
            # --- DETECT DATALOADER TYPE ---
            dataset_type = configs.data_provider.get('dataset_type', 'image_folder')
            
            if dataset_type == 'pk_sampler':
                # --- NEW: BATCH HARD MINING LOOP ---
                for _, (images, labels) in enumerate(self.data_loader['train']):
                    images, labels = images.cuda(), labels.cuda()
                    batch_size = images.shape[0]
                    
                    self.optimizer.zero_grad()
                    
                    # Run model once
                    embeddings_raw = self.model(images)
                    
                    # Normalize
                    s = 32.0
                    embeddings_normalized = F.normalize(embeddings_raw, p=2, dim=1)*s
                    
                    # Loss function does the mining
                    loss = self.criterion(embeddings_normalized, labels)
                    
                    # Standard backward pass
                    loss.backward()
                    
                    # ... (optimizer steps remain the same) ...
                    if configs.backward_config.enable_backward_config:
                        from core.utils.partial_backward import apply_backward_config
                        apply_backward_config(self.model, configs.backward_config)
                    if hasattr(self.optimizer, 'pre_step'):
                        self.optimizer.pre_step(self.model)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    if hasattr(self.optimizer, 'post_step'):
                        self.optimizer.post_step(self.model)

                    train_loss.update(loss, batch_size)
                    
                    t.set_postfix({
                        'loss': train_loss.avg.item(),
                        'batch_size': batch_size,
                        'img_size': images.shape[2],
                        'lr': self.optimizer.param_groups[0]['lr'],
                    })
                    t.update()
                    self.lr_scheduler.step()

            else:
                # --- OLD: TRIPLET LOOP (KEPT FOR COMPATIBILITY) ---
                for _, (anchor_img, positive_img, negative_img) in enumerate(self.data_loader['train']):
                    
                    images_combined = torch.cat((anchor_img, positive_img, negative_img), dim=0).cuda()
                    batch_size = anchor_img.shape[0] 
                    
                    self.optimizer.zero_grad()

                    embeddings_raw = self.model(images_combined)
                    embeddings_normalized = F.normalize(embeddings_raw, p=2, dim=1)
                    
                    emb_anchor = embeddings_normalized[0:batch_size]
                    emb_positive = embeddings_normalized[batch_size : batch_size*2]
                    emb_negative = embeddings_normalized[batch_size*2 : batch_size*3]
                    
                    loss = self.criterion(emb_anchor, emb_positive, emb_negative)

                    loss.backward()

                    if configs.backward_config.enable_backward_config:
                        from core.utils.partial_backward import apply_backward_config
                        apply_backward_config(self.model, configs.backward_config)
                    if hasattr(self.optimizer, 'pre_step'):
                        self.optimizer.pre_step(self.model)
                    self.optimizer.step()
                    if hasattr(self.optimizer, 'post_step'):
                        self.optimizer.post_step(self.model)

                    train_loss.update(loss, batch_size)

                    t.set_postfix({
                        'loss': train_loss.avg.item(),
                        'batch_size': batch_size,
                        'img_size': anchor_img.shape[2],
                        'lr': self.optimizer.param_groups[0]['lr'],
                    })
                    t.update()
                    self.lr_scheduler.step()

        return {
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }