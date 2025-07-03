import wandb
import os
import yaml
from typing import Dict, Any, Optional


class WandbLogger:
    """Weights & Biases logger for CBML training"""
    
    def __init__(self, project_name: str = "cbml-benchmark", 
                 entity: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 enabled: bool = True,
                 save_config: bool = True):
        """
        Initialize wandb logger
        
        Args:
            project_name: Name of the wandb project
            entity: wandb username or team name
            config: Configuration dictionary to log
            enabled: Whether to enable wandb logging
            save_config: Whether to save config as YAML file
        """
        self.enabled = enabled
        self.project_name = project_name
        self.config = config
        self.save_config = save_config
        
        if not enabled:
            return
            
        # Initialize wandb
        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            reinit=True
        )
        
        # Save config as YAML file if enabled
        if save_config and config:
            self._save_config_yaml()
        
    def _save_config_yaml(self):
        """Save config as YAML file in configs directory"""
        if not self.config:
            return
            
        # Create configs directory if it doesn't exist
        configs_dir = "configs"
        os.makedirs(configs_dir, exist_ok=True)
        
        # Generate filename from project name
        # Replace spaces and special characters with underscores
        safe_project_name = self.project_name.replace(" ", "_").replace("-", "_")
        config_filename = f"{safe_project_name}.yaml"
        config_path = os.path.join(configs_dir, config_filename)
        
        # Save config as YAML
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Config saved to: {config_path}")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb"""
        if not self.enabled:
            return
            
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_model(self, model_path: str, aliases: Optional[list] = None):
        """Log model checkpoint to wandb"""
        if not self.enabled:
            return
            
        wandb.save(model_path)
        if aliases:
            wandb.log_model(model_path, aliases=aliases)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to wandb"""
        if not self.enabled:
            return
            
        wandb.config.update(config)
    
    def finish(self):
        """Finish wandb run"""
        if self.enabled:
            wandb.finish()
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch model gradients and parameters"""
        if not self.enabled:
            return
            
        wandb.watch(model, log=log, log_freq=log_freq) 