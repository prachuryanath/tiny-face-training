# an image classification trainer
import os
import sys
import argparse

# an image classification trainer
import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from core.utils.logging import logger
from core.dataset import build_dataset # <-- IMPORT build_pk_sampler
from core.optimizer import build_optimizer
from core.utils.batch_hard_triplet_loss import BatchHardTripletLoss # <-- IMPORT NEW LOSS
from core.utils.arcface import ArcMarginProduct
# from core.trainer.cls_trainer import ClassificationTrainer
import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

from core.trainer.face_rec_trainer import FaceRecognitionTrainer
from core.builder.lr_scheduler import build_lr_scheduler

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--run_dir', type=str, metavar='DIR', help='run directory')
parser.add_argument('--evaluate', action='store_true')


def build_config():  # separate this config requirement so that we can call main() in ray tune
    # support extra args here without setting in args
    args, unknown = parser.parse_known_args()

    load_config_from_file(args.config)
    update_config_from_args(args)
    update_config_from_unknown_args(unknown)


def main():
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert configs.run_dir is not None
    os.makedirs(configs.run_dir, exist_ok=True)
    logger.init()  # dump exp config
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{configs.run_dir}".')

    # set random seed
    torch.manual_seed(configs.manual_seed)
    torch.cuda.manual_seed_all(configs.manual_seed)

    # create dataset
    dataset = build_dataset()
    data_loader = dict()
    
    split = 'train'
    dataset_type = configs.data_provider.get('dataset_type', 'image_folder')
    logger.info("Using standard DistributedSampler for ArcFace.")
    sampler = torch.utils.data.DistributedSampler(
        dataset[split],
        num_replicas=dist.size(),
        rank=dist.rank(),
        seed=configs.manual_seed,
        shuffle=True)
    
    data_loader[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=configs.data_provider.base_batch_size,
        sampler=sampler,
        num_workers=configs.data_provider.n_worker,
        pin_memory=True,
        drop_last=True,
    )
    len_train_loader = len(data_loader[split])

    # create model
    model = build_mcu_model().cuda()

    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()],
            find_unused_parameters=True # May be needed if using backward_config
        )

    metric_fc = None 
    logger.info("Phase 1: ArcFace Training")
    # ArcFace Settings
    embedding_dim = configs.net_config.get('embedding_dim', 128)
    num_classes = configs.data_provider.num_classes
    
    # Initialize ArcFace Head
    metric_fc = ArcMarginProduct(
        in_features=embedding_dim, 
        out_features=num_classes, 
        s=30.0, 
        m=0.50
    ).cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Update both Model and Head
    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': metric_fc.parameters(), 'weight_decay': 5e-4}
    ], lr=configs.run_config.bs256_lr, momentum=0.9)
    lr_scheduler = build_lr_scheduler(optimizer, len_train_loader) # Use correct length

    # --- USE NEW FACE REC TRAINER ---
    trainer = FaceRecognitionTrainer(model, data_loader, criterion, optimizer, lr_scheduler, metric_fc=metric_fc)

    # kick start training
    if configs.resume:
        trainer.resume()  # trying to resume

    if configs.backward_config.enable_backward_config:
        from core.utils.partial_backward import parsed_backward_config, prepare_model_for_backward_config, \
            get_all_conv_ops
        configs.backward_config = parsed_backward_config(configs.backward_config, model)
        prepare_model_for_backward_config(model, configs.backward_config)
        logger.info(f'Getting backward config: {configs.backward_config} \n'
                    f'Total convs {len(get_all_conv_ops(model))}')

    if configs.evaluate:
        val_info_dict = trainer.validate()
        print(val_info_dict)
        return val_info_dict  # for ray tune
    else:
        val_info_dict = trainer.run_training()
        return val_info_dict  # for ray tune


if __name__ == '__main__':
    build_config()
    main()