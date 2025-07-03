# Weights & Biases (wandb) Integration for CBML

This guide explains how to use Weights & Biases (wandb) for experiment tracking in your CBML project with support for multiple datasets.

## Installation

1. Install wandb:
```bash
pip install wandb
```

2. Login to wandb:
```bash
wandb login
```

## 🚀 Quick Start (Recommended)

### Method 1: Simple Project Name (Auto-generate config)

Just provide a project name and dataset - everything is set up automatically:

```bash
# CUB-200-2011 dataset (default)
python scripts/train_with_wandb.py --project "cbml-cub-experiment"

# Cars196 dataset
python scripts/train_with_wandb.py --project "cbml-cars196-experiment" --dataset cars196

# Stanford Online Products (SOP) dataset
python scripts/train_with_wandb.py --project "cbml-sop-experiment" --dataset sop

# Start testing
python scripts/train_with_wandb.py --project "cbml-cub-experiment" --dataset cub --phase test

# With custom wandb entity/team
python scripts/train_with_wandb.py --project "cbml-cars196-experiment" --dataset cars196 --entity "your-team"
```

### Method 2: Using main.py directly

```bash
# Auto-generate config from project name and dataset
python tools/main.py --project "cbml-experiment" --dataset cub --phase train
python tools/main.py --project "cbml-cars196" --dataset cars196 --phase train
python tools/main.py --project "cbml-sop" --dataset sop --phase train

# Use existing config file
python tools/main.py --cfg configs/example_wandb.yaml --phase train
```

## 📊 Supported Datasets

| Dataset | Classes | Default Settings |
|---------|---------|------------------|
| **CUB-200-2011** (`cub`) | 100 | Batch: 60, Iterations: 4000 |
| **Cars196** (`cars196`) | 98 | Batch: 60, Iterations: 4000 |
| **Stanford Online Products** (`sop`) | 11,318 | Batch: 128, Iterations: 8000 |

Each dataset gets optimized configurations automatically:
- **Appropriate batch sizes** based on dataset complexity
- **Correct number of classes** for loss functions
- **Suitable training iterations** for convergence
- **Dataset-specific paths** for train/test splits

## 📁 Auto-Generated Config Files

When you use `--project` and `--dataset`, the system automatically:

1. **Creates config file** at `configs/{project_name}.yaml`
2. **Sets up wandb project** with the same name
3. **Configures dataset-specific settings** (paths, classes, batch sizes)
4. **Optimizes training parameters** for the chosen dataset
5. **Saves output** to `output/{project_name}/`

### Examples:

```bash
# Creates: configs/cbml_cub_resnet50.yaml
python scripts/train_with_wandb.py --project "cbml-cub-resnet50" --dataset cub

# Creates: configs/cbml_cars196_experiment.yaml  
python scripts/train_with_wandb.py --project "cbml-cars196-experiment" --dataset cars196

# Creates: configs/cbml_sop_large_scale.yaml
python scripts/train_with_wandb.py --project "cbml-sop-large-scale" --dataset sop
```

## ⚙️ Manual Configuration (Advanced)

If you want to customize the config manually:

### 1. Enable wandb in your config file

Add the following section to your YAML configuration file:

```yaml
WANDB:
  ENABLED: True
  PROJECT_NAME: "cbml-experiment"  # Your project name
  ENTITY: "your-username"  # Your wandb username or team name
  WATCH_MODEL: True  # Watch model gradients

# Dataset-specific settings
DATA:
  TRAIN_IMG_SOURCE: 'resource/datasets/CUB_200_2011/train.txt'  # or cars196/sop
  TEST_IMG_SOURCE: 'resource/datasets/CUB_200_2011/test.txt'    # or cars196/sop
  TRAIN_BATCHSIZE: 60    # 60 for CUB/Cars196, 128 for SOP
  TEST_BATCHSIZE: 128    # 128 for CUB/Cars196, 256 for SOP

# Loss function classes
LOSSES:
  SOFTTRIPLE_LOSS:
    CLUSTERS: 100        # 100 for CUB, 98 for Cars196, 11318 for SOP
  PROXY_LOSS:
    NB_CLASSES: 100      # 100 for CUB, 98 for Cars196, 11318 for SOP
    SCALING_X: 3         # 3 for CUB/Cars196, 1 for SOP
    SCALING_P: 3         # 3 for CUB/Cars196, 8 for SOP
  CENTER_LOSS:
    CLASS: 100           # 100 for CUB, 98 for Cars196, 11318 for SOP
```

### 2. Example configurations

See `configs/example_wandb.yaml` for a complete example configuration.

## 📊 What Gets Logged

The integration automatically logs:

**Training Metrics:**
- Loss values
- Learning rate
- Memory usage
- Training time per iteration

**Validation Metrics:**
- Recall@1, Recall@2, Recall@4, Recall@8
- Best model performance

**Model Checkpoints:**
- Best model (saved automatically)
- Regular checkpoints (based on CHECKPOINT_PERIOD)

**Configuration:**
- All training parameters
- Model architecture
- Loss function settings
- Dataset information

**System Information:**
- GPU memory usage
- Training time statistics

## 🔧 Customization

### Modifying Auto-Generated Config

After auto-generating a config, you can edit it:

```bash
# Generate config for Cars196
python scripts/train_with_wandb.py --project "my-cars196-experiment" --dataset cars196

# Edit the generated config
vim configs/my_cars196_experiment.yaml

# Run with modified config
python tools/main.py --cfg configs/my_cars196_experiment.yaml --phase train
```

### Adding custom metrics

You can add custom metrics by modifying the trainer:

```python
# In cbml_benchmark/engine/trainer.py
if wandb_logger is not None:
    wandb_logger.log_metrics({
        'custom/metric_name': your_value
    }, step=iteration)
```

### Logging additional data

To log images, plots, or other data:

```python
# Log images
wandb_logger.log_metrics({
    'examples': wandb.Image(your_image)
})

# Log plots
wandb_logger.log_metrics({
    'loss_plot': wandb.plot.line_series(...)
})
```

## 📈 Dashboard Features

Once training starts, you can view:

1. **Real-time metrics** - Live training curves
2. **Model performance** - Validation metrics over time
3. **System monitoring** - GPU usage, memory consumption
4. **Configuration comparison** - Compare different runs
5. **Model artifacts** - Download saved checkpoints

## 🎯 Best Practices

1. **Use descriptive project names** - e.g., "cbml-cub-resnet50-cbml-loss"
2. **Include dataset in project name** - e.g., "cbml-cars196-experiment"
3. **Set meaningful run names** - wandb will auto-generate them
4. **Compare experiments** - Use wandb's comparison features
5. **Save important checkpoints** - They're automatically logged
6. **Monitor system resources** - Watch for memory issues

## 🔍 Troubleshooting

### wandb not logging
- Check if `WANDB.ENABLED = True` in your config
- Verify you're logged in: `wandb login`
- Check internet connection

### Missing metrics
- Ensure the metric is being logged in the trainer
- Check wandb dashboard for any errors

### Configuration not showing
- Verify config is being passed to wandb_logger
- Check that config values are serializable

### Auto-generated config issues
- Check that the project name is valid (no special characters)
- Verify write permissions in the configs directory
- Ensure dataset name is one of: cub, cars196, sop

### Dataset path errors
- Verify your dataset files are in the correct locations:
  - CUB: `resource/datasets/CUB_200_2011/`
  - Cars196: `resource/datasets/cars196/`
  - SOP: `resource/datasets/sop/`

## 📋 Example Workflows

### CUB-200-2011 Experiment
```bash
# Start CUB experiment
python scripts/train_with_wandb.py --project "cbml-cub-resnet50" --dataset cub

# Monitor: https://wandb.ai/your-username/cbml-cub-resnet50
```

### Cars196 Comparison
```bash
# Different loss functions on Cars196
python scripts/train_with_wandb.py --project "cbml-cars196-cbml-loss" --dataset cars196
python scripts/train_with_wandb.py --project "cbml-cars196-triplet-loss" --dataset cars196
# Edit configs to change loss function, then retrain
```

### SOP Large-Scale Training
```bash
# SOP with optimized settings (larger batches, more iterations)
python scripts/train_with_wandb.py --project "cbml-sop-large-scale" --dataset sop

# Monitor: https://wandb.ai/your-username/cbml-sop-large-scale
```

## 🎉 Example Dashboard

After running training, you'll see:
- Training loss curve
- Validation recall metrics per dataset
- Learning rate schedule
- GPU memory usage
- Model checkpoints
- Dataset-specific configuration parameters

Visit your wandb dashboard to explore all the logged data!