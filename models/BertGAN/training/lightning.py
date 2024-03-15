import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
import torch.nn.functional as F
from torch import nn
import math
from models.BertGAN.model.model import Discriminator, Generator
from models.BertGAN.training.configs import TrainingConfig
from models.BertGAN.training.params_schedule import CosineWarmupScheduler


def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


def positional_encoding(shape, device):
    with torch.no_grad(): 
        batch, time_len, var_len, d_model = shape 
        pe = torch.zeros(batch, time_len, d_model, device=device).float()
        position = torch.arange(0, time_len).float().unsqueeze(1)
        
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

    return pe.repeat_interleave(var_len, dim=1)


class VariableEmbedding(nn.Embedding):
    def __init__(self, var_len, embed_size=768):
        super().__init__(var_len, embed_size)


class LeadTimeEmbedding(nn.Embedding):
    def __init__(self, max_lead_time, embed_size=768):
        super().__init__(max_lead_time, embed_size)


class Embedding(nn.Module):
    def __init__(self, max_lead_time, var_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.variable = VariableEmbedding(var_len, embed_size)
        self.time = LeadTimeEmbedding(max_lead_time, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor, variable_seq: torch.Tensor, pos_emb: torch.Tensor, lead_time_seq: torch.Tensor) -> torch.Tensor:
        var_emb = self.variable(variable_seq)
        time_emb = self.time(lead_time_seq)
        return self.dropout(x + var_emb + time_emb + pos_emb)
    

class BertGAN(pl.LightningModule):

    def __init__(self, *, generator: Generator, discriminator: Discriminator, var_list: torch.Tensor,
                 mean_std: torch.Tensor, max_iters: int, config: TrainingConfig, max_var_len=300, max_lead_time=500):
        
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.max_iters = max_iters
        self.mean_std = mean_std
        self.config = config
        self.var_list = var_list
        self.embedding = Embedding(max_lead_time, max_var_len, generator.in_dim, generator.dropout)

        self.automatic_optimization = False
        self.save_hyperparameters()

    
    def setup(self, stage: str) -> None:
        print(stage)
        self.mean_std = self.mean_std.to(self.device)
        self.var_list = self.var_list.to(self.device)


    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)

        self.g_scheduler = CosineWarmupScheduler(
            g_opt, warmup=self.config.warmup_step, max_iters=self.max_iters
        )

        self.d_scheduler = CosineWarmupScheduler(
            d_opt, warmup=self.config.warmup_step, max_iters=self.max_iters
        )
        
        return g_opt, d_opt


    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], mode: str):
        g_opt, d_opt = self.optimizers()

        src = batch[0]
        label = batch[1]
        lead_time = batch[2]

        var_seq = self.var_list.repeat_interleave(src.size(1), dim=0).unsqueeze(0).repeat_interleave(src.size(0), dim=0)
        pe = positional_encoding(src.shape, src.device)

        src = src.view(src.size(0), -1, src.size(-1))
        src_lead_time = lead_time.unsqueeze(1).repeat(1, src.size(1))
        tgt_lead_time = torch.zeros_like(src_lead_time, device=self.device)

        print(src_lead)
     
        src = self.embedding(src, var_seq, pe, src_lead_time) * math.sqrt(self.generator.in_dim)

        # fake.shape = (batch, time * var, hidden)
        fake = self.generator(src)

        ##########################
        # Optimize Discriminator #
        ##########################

        label = label.view(src.size(0), -1, src.size(-1))

        tgt_real = (self.embedding(label, var_seq, pe, tgt_lead_time)) * math.sqrt(self.generator.in_dim)
        tgt_fake = (self.embedding(fake, var_seq, pe, tgt_lead_time)) * math.sqrt(self.generator.in_dim)

        real_label = torch.ones((tgt_real.size(0), tgt_real.size(1)), device=self.device)
        fake_label = torch.zeros((tgt_real.size(0), tgt_real.size(1)), device=self.device)

        err_real = F.binary_cross_entropy_with_logits(
            self.discriminator(tgt_real),
            real_label
        )

        err_fake = F.binary_cross_entropy_with_logits(
            self.discriminator(tgt_fake),
            fake_label
        )
      
        err_d = err_real + err_fake

        err_d.requires_grad_(True)
        
        d_opt.zero_grad()
        self.manual_backward(err_d)
        self.clip_gradients(d_opt, gradient_clip_val=self.config.gradient_clip_val, gradient_clip_algorithm="norm")
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################

        err_g = F.binary_cross_entropy_with_logits(
            self.discriminator(tgt_fake.detach()),
            real_label
        )

        err_g.requires_grad_(True)

        g_opt.zero_grad()
        self.manual_backward(err_g)
        self.clip_gradients(g_opt, gradient_clip_val=self.config.gradient_clip_val, gradient_clip_algorithm="norm")
        g_opt.step()

        self.log_dict({"g_loss": err_g, "d_loss": err_d}, prog_bar=True)

    

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:  # noqa: D102
        self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: D102
        self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:  # noqa: D102
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self(batch)

    def setting(self):
        self.mean_std = self.mean_std.to(self.device)
        self.model.eval()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            src = batch[0].to(self.device)
            delta = batch[2].to(self.device)
            var_seq = batch[3].to(self.device)
            predict = self.model(src, delta, var_seq)
            predict = predict.view(predict.size(0), self.config.time_len, -1, predict.size(-1))

            label = batch[1].to(self.device)
            label = denormalize(label, self.mean_std)
            predict = denormalize(predict, self.mean_std)
            # loss.shape = (batch, time_len, var_len, hidden)
            loss = F.mse_loss(predict, label, reduction='none')
            # loss.shape = (batch, time_len, var_len)
            loss = loss.mean(dim=-1)
        
        return loss, delta
    

