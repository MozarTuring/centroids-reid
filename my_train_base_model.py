import argparse
from config import cfg
from datasets import init_dataset
from modelling.bases import ModelBase
import torch
import sys
import mlflow
import numpy as np
import traceback
import signal
import debugpy
# mlflow.set_tracking_uri("http://0.0.0.0:5000/")
# mlflow.create_experiment("me")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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
        opt_list, _ = self.configure_optimizers() # 会把 lr_schedule 直接赋值给 self
        self.opt, self.opt_center = opt_list
        self.backbone.to(device)
        self.outputs = list()
        self.my_current_epoch = 0 # 不能命名成 current_epoch

    def training_step(self, batch):
        self.mystep += 1
        print(self.my_current_epoch)
        # opt, opt_center = self.optimizers(use_pl_optimizer=True)

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if self.my_current_epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0,
                    float(self.my_current_epoch + 1)
                    / float(self.hparams.SOLVER.WARMUP_EPOCHS),
                )
                for pg in self.opt.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR # 1e-4

        self.opt_center.zero_grad()
        self.opt.zero_grad()

        # for ele in batch:
        #     ele.to(device)
        x, class_labels, camid, isReal = batch  # batch is a tuple
        x = x.to(device)
        class_labels = class_labels.to(device)
        isReal = isReal.to(device)

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
        
        total_loss.backward()
        self.opt.step()

        for param in self.center_loss.parameters():
            param.grad.data *= 1.0 / self.hparams.SOLVER.CENTER_LOSS_WEIGHT
        self.opt_center.step()

        losses = [xent_query, contrastive_loss_query, center_loss] # 训练损失在这里
        losses = [item.detach() for item in losses]
        losses = list(map(float, losses))

        for name, loss_val in zip(self.losses_names, losses): # 注意 losses 长度小于 self.losses_names
            self.losses_dict[name].append(loss_val)
            

        log_data = {
            "step_dist_ap": float(dist_ap.mean()),
            "step_dist_an": float(dist_an.mean()),
        }
        if self.mystep % 206 == 0:
            print("stop")

        self.outputs.append({"loss": total_loss, "other": log_data})

    def training_epoch_end(self,):
        lr = self.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in self.outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in self.outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in self.outputs])

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

        self.outputs = list()
        mlflow.log_metrics(log_data, step=self.my_current_epoch)
        self.my_current_epoch += 1


def term_sig_handler(signum, frame):
    print('catched singal: %d' % signum) # 这个print的内容无法进入 my_log.txt
    mlflow.log_artifact("my_log.txt")
    sys.exit()

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, term_sig_handler)
    parser = argparse.ArgumentParser(description="CLT Model Training")
    parser.add_argument("--debug", action="store_true")
#    parser.add_argument(
#        "--config_file", default="", help="path to config file", type=str
#    )
#    parser.add_argument(
#        "opts",
#        help="Modify config options using the command-line",
#        default=None,
#        nargs=argparse.REMAINDER,
#    )

    args = parser.parse_args()
#    if args.config_file != "":
#        cfg.merge_from_file(args.config_file)
    if args.debug:
        pass
#        debugpy.listen(("0.0.0.0", 55678))
#        print("debugpy is waiting for client")
#        debugpy.wait_for_client()
#        breakpoint()
#    cfg.merge_from_list(args.opts)
    cfg.merge_from_file("configs/256_resnet50.yml")
    cfg.merge_from_list(["GPU_IDS",[0] ,"DATASETS.NAMES",'market1501' ,"DATASETS.ROOT_DIR",'/home/maojingwei/project/resources/dataset' ,"SOLVER.IMS_PER_BATCH",16 ,"TEST.IMS_PER_BATCH",128 ,"SOLVER.BASE_LR",0.00035 ,"OUTPUT_DIR",'' ,"DATALOADER.USE_RESAMPLING",True ,"USE_MIXED_PRECISION",False ,"MODEL.USE_CENTROIDS",False ,"REPRODUCIBLE_NUM_RUNS",1, "DATASETS.train_dir", "bounding_box_train", "DATASETS.query_dir", "letian/query", "DATASETS.gallery_dir", "letian/bounding_box_test"])
    print(cfg)

    mlflow.start_run(run_name="debug", experiment_id="653795851656303629", nested=True)

    dm = init_dataset(
            cfg.DATASETS.NAMES, cfg=cfg, num_workers=cfg.DATALOADER.NUM_WORKERS
        )
    dm.setup()

    ctl_model = CTLModel(
            cfg=cfg,
            num_query=dm.num_query,
            num_classes=dm.num_classes
        )
    ctl_model.to(device)

    # 这里 my_train_dataloader 不能直接跳转，要先去找到dm的定义才能找到怎么进入，这就比较麻烦，会成为一个阻力点。这种通过debug的方式进入就轻松高效很多
    train_loader = dm.my_train_dataloader(
            cfg,
            sampler_name=cfg.DATALOADER.SAMPLER,
            drop_last=cfg.DATALOADER.DROP_LAST,
        )

    val_dataloader = dm.val_dataloader()

    for cur_epoch in range(120):
        ctl_model.train()
        for count, train_batch in enumerate(train_loader):
            # if count > 2 :
            #     break
            ctl_model.training_step(batch=train_batch)
        ctl_model.training_epoch_end()
        ctl_model.lr_scheduler.step()
        
        if cur_epoch in [0,40,80,119]:
            outputs = list()
            print("start eval")
            for eval_batch in val_dataloader:
                ctl_model.backbone.eval()
                ctl_model.bn.eval()
                x, class_labels, camid, idx = eval_batch
                x = x.to(device)

                with torch.no_grad():
                    _, emb = ctl_model.backbone(x)
                    emb = ctl_model.bn(emb)
                outputs.append({"emb": emb, "labels": class_labels, "camid": camid, "idx": idx})
            embeddings = torch.cat([x.pop("emb") for x in outputs]).detach().cpu()
            print(embeddings.shape)
            labels = (
                torch.cat([x.pop("labels") for x in outputs]).detach().cpu().numpy()
            )
            camids = torch.cat([x.pop("camid") for x in outputs]).cpu().detach().numpy()
            ctl_model.my_get_val_metrics(embeddings, labels, camids) # this
#            does not use centroids to retrieval

#    mlflow.log_artifact("my_log.txt")


"""
nohup /home/maojingwei/miniconda3/envs/centroids-reid/bin/python my_train_base_model.py \
MLFLOW.RUN_NAME test > my_log.txt 2>&1 &


mlflow ui --host 0.0.0.0
"""
