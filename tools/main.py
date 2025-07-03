import argparse
import torch
import os
import yaml
from typing import Optional

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss,build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import build_lr_scheduler, build_optimizer
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.wandb_logger import WandbLogger


def create_config_from_project(project_name: str, dataset: str = "cub", base_config: Optional[dict] = None) -> str:
    """
    Create a config file based on project name and dataset, return the path
    
    Args:
        project_name: Name of the wandb project
        dataset: Dataset name (cub, cars196, sop)
        base_config: Base configuration to use (optional)
    
    Returns:
        Path to the created config file
    """
    # Create configs directory if it doesn't exist
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    # Generate safe filename from project name
    safe_project_name = project_name.replace(" ", "_").replace("-", "_")
    config_filename = f"{safe_project_name}.yaml"
    config_path = os.path.join(configs_dir, config_filename)
    
    # If config file already exists, use it
    if os.path.exists(config_path):
        print(f"Using existing config: {config_path}")
        return config_path
    
    # Dataset-specific configurations
    dataset_configs = {
        "cub": {
            "train_source": "resource/datasets/CUB_200_2011/train.txt",
            "test_source": "resource/datasets/CUB_200_2011/test.txt",
            "num_classes": 100,
            "max_iters": 4000,
            "train_batchsize": 60,
            "test_batchsize": 128,
            "num_instances": 5
        },
        "cars196": {
            "train_source": "resource/datasets/cars196/train.txt",
            "test_source": "resource/datasets/cars196/test.txt", 
            "num_classes": 98,
            "max_iters": 4000,
            "train_batchsize": 60,
            "test_batchsize": 128,
            "num_instances": 5
        },
        "sop": {
            "train_source": "resource/datasets/sop/train.txt",
            "test_source": "resource/datasets/sop/test.txt",
            "num_classes": 11318,
            "max_iters": 8000,
            "train_batchsize": 128,
            "test_batchsize": 256,
            "num_instances": 4
        }
    }
    
    # Get dataset config
    if dataset.lower() not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: {list(dataset_configs.keys())}")
    
    dataset_cfg = dataset_configs[dataset.lower()]
    
    # Create default config if none provided
    if base_config is None:
        base_config = {
            'WANDB': {
                'ENABLED': True,
                'PROJECT_NAME': project_name,
                'ENTITY': None,
                'WATCH_MODEL': True
            },
            'MODEL': {
                'DEVICE': 'cuda',
                'BACKBONE': {
                    'NAME': 'resnet50'
                },
                'PRETRAIN': 'imagenet',
                'HEAD': {
                    'NAME': 'linear_norm',
                    'DIM': 512
                }
            },
            'LOSSES': {
                'NAME': 'cbml_loss',
                'NAME_AUX': '',
                'AUX_WEIGHT': 0.01,
                # Dataset-specific loss configurations
                'SOFTTRIPLE_LOSS': {
                    'CLUSTERS': dataset_cfg["num_classes"]
                },
                'PROXY_LOSS': {
                    'NB_CLASSES': dataset_cfg["num_classes"],
                    'SCALING_X': 3 if dataset.lower() != 'sop' else 1,
                    'SCALING_P': 3 if dataset.lower() != 'sop' else 8
                },
                'CENTER_LOSS': {
                    'CLASS': dataset_cfg["num_classes"]
                }
            },
            'DATA': {
                'TRAIN_IMG_SOURCE': dataset_cfg["train_source"],
                'TEST_IMG_SOURCE': dataset_cfg["test_source"],
                'TRAIN_BATCHSIZE': dataset_cfg["train_batchsize"],
                'TEST_BATCHSIZE': dataset_cfg["test_batchsize"],
                'NUM_WORKERS': 8,
                'NUM_INSTANCES': dataset_cfg["num_instances"],
                'KNN': 1
            },
            'SOLVER': {
                'MAX_ITERS': dataset_cfg["max_iters"],
                'BASE_LR': 0.01,
                'WEIGHT_DECAY': 0.0005,
                'MOMENTUM': 0.9,
                'CHECKPOINT_PERIOD': 200
            },
            'SAVE_DIR': f'output/{safe_project_name}',
            'VALIDATION': {
                'VERBOSE': 200,
                'IS_VALIDATION': True
            }
        }
    
    # Save config as YAML
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, indent=2)
    
    print(f"Created config file: {config_path}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Classes: {dataset_cfg['num_classes']}")
    print(f"Max iterations: {dataset_cfg['max_iters']}")
    return config_path


def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    
    # Initialize wandb logger
    wandb_logger = None
    if hasattr(cfg, 'WANDB') and cfg.WANDB.ENABLED:
        # Convert config to dict for wandb
        config_dict = {}
        for key, value in cfg.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        wandb_logger = WandbLogger(
            project_name=cfg.WANDB.PROJECT_NAME,
            entity=cfg.WANDB.ENTITY if hasattr(cfg.WANDB, 'ENTITY') else None,
            config=config_dict,
            enabled=cfg.WANDB.ENABLED,
            save_config=False  # We're handling config saving separately
        )
        
        # Watch model gradients
        if hasattr(cfg.WANDB, 'WATCH_MODEL') and cfg.WANDB.WATCH_MODEL:
            model = build_model(cfg)
            wandb_logger.watch(model, log="gradients", log_freq=100)
    
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)
    criterion_aux = None
    if cfg.LOSSES.NAME_AUX != '':
        criterion_aux = build_aux_loss(cfg)

    loss_param = None
    if cfg.LOSSES.NAME == 'softtriple_loss' or cfg.LOSSES.NAME == 'proxynca_loss' or cfg.LOSSES.NAME == 'center_loss' or cfg.LOSSES.NAME == 'adv_loss':
        loss_param = criterion
    if cfg.LOSSES.NAME_AUX == 'softtriple_loss' or cfg.LOSSES.NAME_AUX == 'proxynca_loss' or cfg.LOSSES.NAME_AUX == 'center_loss' or cfg.LOSSES.NAME_AUX == 'adv_loss':
        loss_param = criterion_aux

    optimizer = build_optimizer(cfg, model,loss_param=loss_param)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.SAVE_DIR)

    try:
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            criterion_aux,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            logger,
            wandb_logger
        )
    finally:
        # Ensure wandb run is finished
        if wandb_logger is not None:
            wandb_logger.finish()

def test(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    
    # Initialize wandb logger for testing
    wandb_logger = None
    if hasattr(cfg, 'WANDB') and cfg.WANDB.ENABLED:
        wandb_logger = WandbLogger(
            project_name=cfg.WANDB.PROJECT_NAME,
            entity=cfg.WANDB.ENTITY if hasattr(cfg.WANDB, 'ENTITY') else None,
            config={},
            enabled=cfg.WANDB.ENABLED
        )
    
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    val_loader = build_data(cfg, is_train=False)
    logger.info(val_loader.dataset)

    try:
        do_test(
            model,
            val_loader,
            logger,
            wandb_logger
        )
    finally:
        # Ensure wandb run is finished
        if wandb_logger is not None:
            wandb_logger.finish()


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file (optional if using --project)',
        default=None,
        type=str)
    parser.add_argument(
        '--project',
        dest='project_name',
        help='wandb project name (will auto-generate config)',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        choices=['cub', 'cars196', 'sop'],
        default='cub',
        help='dataset to use (cub, cars196, sop)')
    parser.add_argument(
        '--phase',
        dest='train_test',
        help='train or test',
        default='train',
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # If project name is provided, auto-generate config
    if args.project_name:
        config_path = create_config_from_project(args.project_name, args.dataset)
        cfg.merge_from_file(config_path)
    elif args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    else:
        raise ValueError("Either --cfg or --project must be provided")
    
    if args.train_test == 'train':
        train(cfg)
    else:
        test(cfg)
