# encoding: utf-8
"""
Adapted and extended by:
@author: mikwieczorek
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities import AttributeDict, rank_zero_only
from torch import tensor
from tqdm import tqdm

from config import cfg
from modelling.bases import ModelBase
from utils.misc import run_main


class CTLModel(ModelBase):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.losses_names = [
            "query_xent",
            "query_triplet",
            "query_center",
            "centroid_triplet",
        ]
        self.losses_dict = {n: [] for n in self.losses_names}
        self.mystep = 0

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.mystep += 1
        opt, opt_center = self.optimizers(use_pl_optimizer=True)

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if self.trainer.current_epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0,
                    float(self.trainer.current_epoch + 1)
                    / float(self.hparams.SOLVER.WARMUP_EPOCHS),
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR

        opt_center.zero_grad()
        opt.zero_grad()

        x, class_labels, camid, isReal = batch  # batch is a tuple

        # Get backbone features
        _, features = self.backbone(x)

        # query
        contrastive_loss_query, dist_ap, dist_an = self.contrastive_loss(
            features, class_labels, mask=isReal
        )
        contrastive_loss_query = (
            contrastive_loss_query * self.hparams.SOLVER.QUERY_CONTRASTIVE_WEIGHT
        )

        center_loss = self.hparams.SOLVER.CENTER_LOSS_WEIGHT * self.center_loss(
            features, class_labels
        )
        bn_features = self.bn(features)
        cls_score = self.fc_query(bn_features)
        xent_query = self.xent(cls_score, class_labels)
        xent_query = xent_query * self.hparams.SOLVER.QUERY_XENT_WEIGHT

        total_loss = center_loss + xent_query + contrastive_loss_query

        self.manual_backward(total_loss, optimizer=opt)

        opt.step()

        for param in self.center_loss.parameters():
            param.grad.data *= 1.0 / self.hparams.SOLVER.CENTER_LOSS_WEIGHT
        opt_center.step()

        losses = [xent_query, contrastive_loss_query, center_loss] # 训练损失在这里
        losses = [item.detach() for item in losses]
        losses = list(map(float, losses))

        for name, loss_val in zip(self.losses_names, losses):
            self.losses_dict[name].append(loss_val)

        log_data = {
            "step_dist_ap": float(dist_ap.mean()),
            "step_dist_an": float(dist_an.mean()),
        }
        if self.mystep % 206 == 0:
            print("stop")

        return {"loss": total_loss, "other": log_data}

    def training_epoch_end(self, outputs):
        # 会先执行 valuation
        if hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch + 1)

        lr = self.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
        }

        if hasattr(self, "losses_dict"):
            for name, loss_val in self.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                self.losses_dict[name] = []  ## Zeroing values after a completed epoch

        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)
        self.trainer.accelerator_backend.barrier()


if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="CLT Model Training")
#    parser.add_argument(
#        "--config_file", default="", help="path to config file", type=str
#    )
#    parser.add_argument(
#        "opts",
#        help="Modify config options using the command-line",
#        default=None,
#        nargs=argparse.REMAINDER,
#    )

#    args = parser.parse_args()
    

#    if args.config_file != "":
#        cfg.merge_from_file(args.config_file)
#    print(args.opts)
#    cfg.merge_from_list(args.opts)
#    print(cfg)
    cfg.merge_from_file("configs/256_resnet50.yml")
    cfg.merge_from_list(["GPU_IDS",[0] ,"DATASETS.NAMES",'market1501' ,"DATASETS.ROOT_DIR",'/home/maojingwei/project/resources/dataset' ,"SOLVER.IMS_PER_BATCH",16 ,"TEST.IMS_PER_BATCH",128 ,"SOLVER.BASE_LR",0.00035 ,"OUTPUT_DIR",'expr20230504' ,"DATALOADER.USE_RESAMPLING",True ,"USE_MIXED_PRECISION",False ,"MODEL.USE_CENTROIDS",False ,"REPRODUCIBLE_NUM_RUNS",1,"TEST.ONLY_TEST",True])
    print(cfg)

    logger_save_dir = f"{Path(__file__).stem}"
    print(logger_save_dir)

    run_main(cfg, CTLModel, logger_save_dir)



"""
nohup python train_base_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/home/maojingwei/project/resources/dataset' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR 'logs/market1501/256_resnet50_base' \
DATALOADER.USE_RESAMPLING True \
USE_MIXED_PRECISION False \
MODEL.USE_CENTROIDS False \
REPRODUCIBLE_NUM_RUNS 1 > my_log.txt 2>&1 &

"""
