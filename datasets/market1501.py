# encoding: utf-8
"""
Partially based on work by:
@author:  sherlock
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import glob
import os.path as osp
import re
from collections import defaultdict

import pytorch_lightning as pl
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              SequentialSampler)

from .bases import (BaseDatasetLabelled, BaseDatasetLabelledPerPid,
                    ReidBaseDataModule, collate_fn_alternative, pil_loader)
from .samplers import get_sampler
from .transforms import ReidTransforms


class Market1501(ReidBaseDataModule):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)

    Version that will not supply resampled instances
    """
    dataset_dir = 'market1501'

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, cfg.DATASETS.train_dir)
        self.query_dir = osp.join(self.dataset_dir, cfg.DATASETS.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, cfg.DATASETS.gallery_dir)




    def setup(self):
        self._check_before_run()
        transforms_base = ReidTransforms(self.cfg)
        
        train, train_dict, train_pid_set = self._process_dir(self.train_dir, relabel=True)
        self.train_dict = train_dict
        self.train_list = train # 似乎没有用到
        self.train = BaseDatasetLabelledPerPid(train_dict, transforms_base.build_transforms(is_train=True), self.num_instances, self.cfg.DATALOADER.USE_RESAMPLING) # 这个是 dataloader 的输入

        query, query_dict, query_pid_set = self._process_dir(self.query_dir, relabel=False)
        gallery, gallery_dict, gallery_pid_set  = self._process_dir(self.gallery_dir, relabel=False)
        self.query_list = query
        self.gallery_list = gallery
        self.val = BaseDatasetLabelled(query+gallery, transforms_base.build_transforms(is_train=False))

        self._print_dataset_statistics(train, query, gallery)
        # For reid_metic to evaluate properly
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(query)
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)
        self.num_query = len(query)
        self.num_classes = num_train_pids

    def _process_dir(self, dir_path, relabel=False):
        my_pid_set = set()
        img_paths = glob.glob(osp.join(dir_path, '*.jpg')) # 这个很方便，以后可以用起来
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            my_pid_set.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset_dict = defaultdict(list)
        dataset = []

        for idx, img_path in enumerate(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups()) # camid 摄像头id
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, idx))
            dataset_dict[pid].append((img_path, pid, camid, idx))

        return dataset, dataset_dict, my_pid_set
