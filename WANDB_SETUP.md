# Weights & Biases (wandb) Integration for CBML

This guide explains how to use Weights & Biases (wandb) for experiment tracking in your CBML project.

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
# Start training with CUB dataset (default)
python scripts/train_with_wandb.py --project "cbml-cub-experiment"

# Start training with Cars196 dataset
python scripts/train_with_wandb.py --project "cbml-cars196-experiment" --dataset cars196

# Start training with Stanford Online Products dataset
python scripts/train_with_wandb.py --project "cbml-sop-experiment" --dataset sop

# Start testing
python scripts/train_with_wandb.py --project "cbml-cub-experiment" --dataset cub --phase test

# With custom wandb entity/team
python scripts/train_with_wandb.py --project "cbml-cub-experiment" --dataset cub --entity "your-team"
```

### Method 2: Using main.py directly

```bash
# Auto-generate config from project name with dataset
python tools/main.py --project "cbml-cub-experiment" --dataset cub --phase train
python tools/main.py --project "cbml-cars196-experiment" --dataset cars196 --phase train
python tools/main.py --project "cbml-sop-experiment" --dataset sop --phase train

# Use existing config file
python tools/main.py --cfg configs/example_wandb.yaml --phase train
```

## 📊 Supported Datasets

The system supports three major datasets for deep metric learning:

### 1. CUB-200-2011 (Default)
- **Description**: Caltech-UCSD Birds-200-2011 dataset
- **Classes**: 200 bird species
- **Images**: ~11,800 images
- **Usage**: `--dataset cub` (default)

### 2. Cars196
- **Description**: Cars dataset with 196 car models
- **Classes**: 196 car models
- **Images**: ~16,185 images
- **Usage**: `--dataset cars196`

### 3. Stanford Online Products (SOP)
- **Description**: E-commerce product dataset
- **Classes**: 22,634 product categories
- **Images**: ~120,053 images
- **Usage**: `--dataset sop`

## 📁 Auto-Generated Config Files

When you use `--project` with `--dataset`, the system automatically:

1. **Creates config file** at `configs/{project_name}.yaml`
2. **Sets up wandb project** with the same name
3. **Configures dataset-specific settings** (paths, batch sizes, etc.)
4. **Saves output** to `output/{project_name}/`

Example: `--project "cbml-cars196-resnet50" --dataset cars196` creates:
- Config: `configs/cbml_cars196_resnet50.yaml`
- Output: `output/cbml_cars196_resnet50/`
- Wandb project: `cbml-cars196-resnet50`
- Dataset paths: Cars196 dataset paths

## 🗂️ Dataset Preparation

### CUB-200-2011
```bash
./scripts/prepare_cub.sh
```

### Cars196
```bash
./scripts/prepare_cars196.sh
```

### Stanford Online Products
```bash
./scripts/prepare_sop.sh
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
```

### 2. Example configurations

- `configs/example_wandb.yaml` - Complete example with wandb
- `configs/example_cars196.yaml` - Cars196 dataset example
- `configs/example_sop.yaml` - Stanford Online Products example

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
6. **Dataset tracking** - Which dataset was used

## 🎯 Best Practices

1. **Use descriptive project names** - e.g., "cbml-cars196-resnet50-cbml-loss"
2. **Include dataset in project name** - Makes it easier to organize experiments
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

### Dataset issues
- Ensure dataset is prepared using the preparation scripts
- Check that dataset paths exist in the config
- Verify train.txt and test.txt files are generated

## 📋 Example Workflow

```bash
# 1. Login to wandb
wandb login

# 2. Prepare dataset (choose one)
./scripts/prepare_cub.sh
# OR
./scripts/prepare_cars196.sh
# OR
./scripts/prepare_sop.sh

# 3. Start training with auto-generated config
python scripts/train_with_wandb.py --project "cbml-cars196-resnet50" --dataset cars196

# 4. Monitor on wandb dashboard
# Visit: https://wandb.ai/your-username/cbml-cars196-resnet50

# 5. Compare with different settings
python scripts/train_with_wandb.py --project "cbml-cars196-resnet50-lr001" --dataset cars196
# Edit configs/cbml_cars196_resnet50_lr001.yaml to change learning rate
python tools/main.py --cfg configs/cbml_cars196_resnet50_lr001.yaml --phase train
```

## 🎉 Example Dashboard

After running training, you'll see:
- Training loss curve
- Validation recall metrics
- Learning rate schedule
- GPU memory usage
- Model checkpoints
- Configuration parameters
- Dataset information

Visit your wandb dashboard to explore all the logged data! 