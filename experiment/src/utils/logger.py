import wandb
import datetime
import yaml
from typing import Dict, Any
import os
        
class WandbLogger():
    def __init__(self, config: dict[str, any]) -> None:
        self.init_wandb(config)

    @staticmethod
    def init_wandb(config: dict[str, Any]) -> None:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        model_name = f"Sparse_{config['model']['name']}" if config['model']['Sparse'] else config['model']['name']
        experiment = config['wandb']['experiment']
        project_name = "trogan detection"
        
        run_name = f"{experiment}_{current_date}"
        
        # wandb config 
        wandb_config = {
            "model_name": config['model']['name'],
            "batch_size": config['data']['train']['batch_size'],
            "learning_rate": config['train']['optimizer']['config']['lr'],
            "optimizer": config['train']['optimizer']['name'],
            "criterion": config['train']['criterion']['name'],
            "lr_scheduler": config['train']['lr_scheduler']['name'],
            "learning_rate": config['train']['optimizer']['config']['lr'],
            "max_node": config['data']['max_node'],
            "SAM": config['train']['SAM'],
            "patient": config['train']['early_stopping']['patient']
        }
        
        # wandb initialize 
        try:
            wandb.init(project=project_name, config=wandb_config, name=run_name)
            run_id = wandb.run.id
            config['wandb']['run_id'] = run_id
        except Exception as e:
            print(f"Error during W&B initialization: {e}")

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float = None, val_metric: float = None) -> None:
        metrics = {
            "train_loss": train_loss,
        }
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if val_metric is not None:
            metrics["val_metric"] = val_metric
        
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"Error recording W&B logs: {e}")

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float = None, val_acc: float = None, val_f1: float = None, val_precision: float = None, val_recall: float = None) -> None:
        metrics = {
            "train_loss": train_loss,
        }
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if val_f1 is not None:
            metrics["val_f1_score"] = val_f1
        if val_precision is not None:
            metrics["val_precision"] = val_precision
        if val_recall is not None:
            metrics["val_recall"] = val_recall
        if val_acc is not None:
            metrics["val_acc"] = val_acc
        
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"Error recording W&B logs: {e}")
        
    def finish_wandb(self) -> None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Error during wandb finish: {e}")
                
def save_config(config: Dict[str, Any], output_dir: str, dev: bool=False) -> None:
    if dev:
        timestamp = 'dev'
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    folder_name = f"{timestamp}_{config['model']['name']}_{config['developer']}"
    folder_path = os.path.join(output_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    
    output_path = os.path.join(folder_path, 'config.yaml')
    
    config['paths']['output_dir'] = folder_path
    
    with open(output_path, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Config file saved to {output_path}")