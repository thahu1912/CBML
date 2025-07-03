import argparse
import torch
import yaml
import os
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
    Create a config file based on project name and dataset
    
    Args:
        project_name: Name of the wandb project
        dataset: Dataset name ('cub', 'cars196', 'sop')
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
    
    # Dataset configurations
    dataset_configs = {
        "cub": {
            "TRAIN_IMG_SOURCE": "resource/datasets/CUB_200_2011/train.txt",
            "TEST_IMG_SOURCE": "resource/datasets/CUB_200_2011/test.txt",
            "TRAIN_BATCHSIZE": 60,
            "TEST_BATCHSIZE": 128,
            "NUM_WORKERS": 8,
            "NUM_INSTANCES": 5,
            "KNN": 1
        },
        "cars196": {
            "TRAIN_IMG_SOURCE": "resource/datasets/Cars196/train.txt",
            "TEST_IMG_SOURCE": "resource/datasets/Cars196/test.txt",
            "TRAIN_BATCHSIZE": 60,
            "TEST_BATCHSIZE": 128,
            "NUM_WORKERS": 8,
            "NUM_INSTANCES": 5,
            "KNN": 1
        },
        "sop": {
            "TRAIN_IMG_SOURCE": "resource/datasets/Stanford_Online_Products/train.txt",
            "TEST_IMG_SOURCE": "resource/datasets/Stanford_Online_Products/test.txt",
            "TRAIN_BATCHSIZE": 60,
            "TEST_BATCHSIZE": 128,
            "NUM_WORKERS": 8,
            "NUM_INSTANCES": 5,
            "KNN": 1
        }
    }
    
    # Validate dataset
    if dataset not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {list(dataset_configs.keys())}")
    
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
                'AUX_WEIGHT': 0.01
            },
            'DATA': dataset_configs[dataset],
            'SOLVER': {
                'MAX_ITERS': 4000,
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
    return config_path


def train(cfg, config_file_path=None):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    
    # Initialize wandb logger automatically when using --config
    wandb_logger = None
    if config_file_path:
        # Extract project name from config filename
        config_filename = os.path.basename(config_file_path)
        project_name = config_filename.replace('.yaml', '').replace('example_', 'cbml-')
        
        # Convert config to dict for wandb
        config_dict = {}
        for key, value in cfg.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        wandb_logger = WandbLogger(
            project_name=project_name,
            entity=None,
            config=config_dict,
            enabled=True,
            save_config=False
        )
        
        # Watch model gradients
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

def test(cfg, config_file_path=None):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    
    # Initialize wandb logger for testing when using --config
    wandb_logger = None
    if config_file_path:
        # Extract project name from config filename
        config_filename = os.path.basename(config_file_path)
        project_name = config_filename.replace('.yaml', '').replace('example_', 'cbml-')
        
        wandb_logger = WandbLogger(
            project_name=project_name,
            entity=None,
            config={},
            enabled=True
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
        '--config',
        dest='cfg_file',
        help='config file path',
        default=None,
        type=str)
    parser.add_argument(
        '--cfg',
        dest='cfg_file_alt',
        help='config file path (alternative to --config)',
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
    
    # Handle config file argument (support both --config and --cfg)
    config_file = args.cfg_file or args.cfg_file_alt
    
    # If project name is provided, auto-generate config
    if args.project_name:
        config_path = create_config_from_project(args.project_name, args.dataset)
        cfg.merge_from_file(config_path)
    elif config_file:
        cfg.merge_from_file(config_file)
    else:
        raise ValueError("Either --config/--cfg or --project must be provided")
    
    if args.train_test == 'train':
        train(cfg, config_file)
    else:
        test(cfg, config_file)
