import logging
import pytorch_lightning as pl
import torch
import yaml
from typing import List
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from absl import app
from pytorch_lightning.utilities.model_summary import ModelSummary

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from datasets.weather_bench import WeatherDataset
from models.VariableEncoder.datasets.dataset import CustomDataset
from models.VariableEncoder.models.model import VariableEncoder

from models.VariableEncoder.training.configs import TrainingConfig
from models.VariableEncoder.training.configs import TrainingRunConfig
from models.VariableEncoder.training.lightning import TrainModule


def get_normal_dataset(config: TrainingConfig, year_offset: int, tgt_time_len: int):
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(year_offset, device=device)
    # dataset.shape:  torch.Size([7309, 100, 1450])
    source, label, mean_std = weather.load(config.air_variable, config.surface_variable)

    dataset = CustomDataset(source, label, tgt_time_len)
    return dataset, source.shape, mean_std, label.size(-1)


        
def _main(args) -> None:
    config_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'configs/train_config.yaml')
    model_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))), 'checkpoints/dVAE')
    
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config: TrainingRunConfig = TrainingRunConfig.parse_obj(config_dict)
    except FileNotFoundError:
        logging.error(f"Config file {config_path} does not exist. Exiting.")
    except yaml.YAMLError:
        logging.error(f"Config file {config_path} is not valid YAML. Exiting.")
    except ValidationError as e:
        logging.error(f"Config file {config_path} is not valid. Exiting.\n{e}")
    else:
        pl.seed_everything(config.seed)

        train_offset = config.training.train_offset
        tgt_time_len = 1 * config.training.tgt_time_len

        # shape = (time, var, hidden)
        dataset_info, dataset, shape, mean_std, out_dim = get_normal_dataset(,train_offset, tgt_time_len)

        print("DATASET SHAPE: " , shape)

        logger = WandbLogger(save_dir=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'tb_logs'), name="my_model")
        model = VariableEncoder(
            var_len=shape[1],
            tgt_time_len=tgt_time_len,
            dim_model=shape[2],
            out_dim=out_dim,
            batch_size=config.training.batch_size,
            num_heads=config.model.num_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            dropout=config.model.dropout
        )
        
        # Use a custom dataset class with proper transformations

        train_ds, test_ds = torch.utils.data.random_split(
            dataset,
            [0.8, 0.2],
        )
        val_ds, test_ds = torch.utils.data.random_split(
            test_ds,
            [0.3, 0.7],
        )   

        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=12
        )
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, drop_last=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, drop_last=True, num_workers=8)

        print("setting lr rate: ", config.training.learning_rate)

        model_pl = TrainModule(model=model, mean_std=mean_std, var_len=shape[1], 
                               max_iters=config.training.max_epochs*len(train_loader), 
                               var_lv=dataset_info[0], var_nlv=dataset_info[1], levels=dataset_info[2], config=config.training)
        summary = ModelSummary(model_pl, max_depth=-1)
        print(summary)

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
            
            ],
            precision="bf16-mixed"
        )

        trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model_pl, dataloaders=test_loader)


if __name__ == "__main__":
    app.run(_main)
