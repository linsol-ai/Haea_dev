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
from absl import flags


import sys,os
from typing import Tuple
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from datasets.weather_bench import WeatherDataset, VariableVocab
from models.HAEA.datasets.denoised_dataset import DenoisingDataset, TimeVocab
from models.HAEA.models.model import Haea

from models.HAEA.training.configs import TrainingConfig
from models.HAEA.training.configs import TrainingRunConfig
from models.HAEA.training.lightning import TrainModule

import os
env_cp = os.environ.copy()


FLAGS = flags.FLAGS
WORLD_SIZE = flags.DEFINE_integer('WORLD_SIZE', None, help='define gpu size')
flags.mark_flag_as_required("WORLD_SIZE")

if 'NODE_RANK' in env_cp.keys():
    node_rank, local_rank, world_size = int(env_cp['NODE_RANK']), int(env_cp['LOCAL_RANK']), int(env_cp['WORLD_SIZE'])
else:
    world_size = 1
    local_rank = 0

config_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'configs/train_config.yaml')


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


def split_datetime_range(start, end, n):
    # 시작과 끝 사이의 총 시간을 계산
    total_duration = end - start
    # 각 구간의 길이를 계산
    interval_duration = total_duration / n
    # 각 구간의 시작점을 저장할 리스트 생성
    intervals = [start + interval_duration * i for i in range(n)]
    # 마지막 구간의 끝을 추가
    intervals.append(end)
    return intervals


def get_normal_dataset(config: TrainingConfig) -> Tuple[DenoisingDataset, torch.Tensor, VariableVocab, TimeVocab]:

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    vars = config.air_variable + config.surface_variable + config.only_input_variable + config.constant_variable

    intervals = split_datetime_range(config.train_start, config.train_end, FLAGS.WORLD_SIZE)

    weather = WeatherDataset(intervals[local_rank], intervals[local_rank+1], device=device, download_variables=vars, download_levels=config.levels)
    # dataset.shape:  torch.Size([7309, 100, 1450])
    
    source, mean_std, var_vocab = weather.load_one(config.air_variable, config.surface_variable, config.only_input_variable, 
                                        config.constant_variable, level=config.levels)
    src_var_list = var_vocab.get_code(vars)
    tgt_var_list = var_vocab.get_code(config.air_variable + config.surface_variable)

    time_vocab = TimeVocab(source, src_var_list, tgt_var_list, config.time_len)
    
    dataset = DenoisingDataset(
        time_vocab,
        127,
        mask=config.mask_ratio,
        mask_random=0,
        insert=0,
        rotate=0,
        permute_sentences=0,
        replace_length=-1,
        mask_length='span-poisson',
        poisson_lambda=3
    )
    return dataset, mean_std, var_vocab, time_vocab


class DataModule(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.dataset, self.mean_std, self.var_vocab, self.time_vocab = get_normal_dataset(self.config)
        train_ds, test_ds = torch.utils.data.random_split(
            self.dataset,
            [0.8, 0.2],
        )
        val_ds, test_ds = torch.utils.data.random_split(
            test_ds,
            [0.3, 0.7],
        )
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size, drop_last=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.batch_size, drop_last=True, num_workers=2)

def main(argv):
    data_module = DataModule(config.training)
    dataset = data_module.dataset
    mean_std = data_module.mean_std
    max_iters = config.training.max_epochs*(len(data_module.train_ds) // config.training.batch_size)
    print(f"max_iters: {max_iters}")
    print(len(data_module.var_vocab))

    logger = WandbLogger(save_dir=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'tb_logs'), name="my_model")
    model = Haea(
        in_dim=data_module.time_vocab.dataset.size(-1),
        out_dim=data_module.time_vocab.dataset.size(-1),
        tgt_mask=data_module.time_vocab.tgt_mask,
        src_var_list=data_module.time_vocab.src_var_list,
        
        num_heads=config.model.num_heads,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        dropout=config.model.dropout,
        max_var_len=len(data_module.var_vocab)
    )

    print("setting lr rate: ", config.training.learning_rate)

    model_pl = TrainModule(model=model, mean_std=mean_std, max_iters=max_iters, 
                        config=config.training)
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=FLAGS.WORLD_SIZE,
        strategy="ddp",
        max_epochs=config.training.max_epochs,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=[
        LearningRateMonitor(logging_interval="step"),
        ],
        precision="bf16-mixed"
    )

    trainer.fit(model_pl, datamodule=data_module)

if __name__ == '__main__':
  app.run(main)
