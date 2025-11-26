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
# try:
#     from ..utils.verification import load_bin, test
# except ImportError:
#     print("Warning: algorithm.verification could not be imported. Validation will fail.")
#     print("Please ensure 'verification.py' is in the 'algorithm/' directory.")
#     load_bin = None
#     test = None
load_bin = None

class FaceRecognitionTrainer(BaseTrainer):

    def __init__(self, model, data_loader, criterion, optimizer, lr_scheduler, metric_fc=None):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)
        self.metric_fc = metric_fc

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
        if self.metric_fc:
            self.metric_fc.train() # Ensure Head is in train mode
        
        self.data_loader['train'].sampler.set_epoch(epoch)
        dataset_type = configs.data_provider.get('dataset_type', 'image_folder')    

        train_loss = DistributedMetric('train_loss')

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            
            dataset_type = configs.data_provider.get('dataset_type', 'image_folder')
            
            # --- PHASE 1: ARCFACE TRAINING LOOP ---
            for _, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                batch_size = images.shape[0]
                
                self.optimizer.zero_grad()
                
                # 1. Get Embeddings from Backbone (Floating Point)
                embeddings = self.model(images) 
                
                # 2. Pass Embeddings + Labels to ArcFace Head
                # This calculates the angular margin logits
                output = self.metric_fc(embeddings, labels)
                
                # 3. Calculate CrossEntropy Loss on the angular logits
                loss = self.criterion(output, labels)
                
                loss.backward()
                
                if configs.backward_config.enable_backward_config:
                    from core.utils.partial_backward import apply_backward_config
                    apply_backward_config(self.model, configs.backward_config)

                if hasattr(self.optimizer, 'pre_step'):
                    self.optimizer.pre_step(self.model)
                
                # Clip gradients for stability (important for ArcFace)
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
        return {
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }