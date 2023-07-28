import os
import sys
import argparse
import yaml
import shutil
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from trainer import SynthesisTask


parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="workspace/config", type=str)
parser.add_argument("--gpus", type=str, default="0,1")
parser.add_argument("--port", type=str, default="12345")
args = parser.parse_args()

# Load config yaml file and pre-process params
# merge params_default.yaml | params_{dataset}.yaml | extra_config
default_config_path = os.path.join(args.config_path, "params_default.yaml")
with open(default_config_path, "r") as f:
    config = yaml.safe_load(f)

with open(os.path.join(args.config_path, "params_coco.yaml"), "r") as f:
    dataset_specific_config = yaml.safe_load(f)
    config.update(dataset_specific_config)

# Pre-process args
config["lr.decay_steps"] = [int(s) for s in str(config["lr.decay_steps"]).split(",")]
config["current_epoch"] = 0

# Config gpu
if args.gpus == "all":
    world_size = torch.cuda.device_count()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(str(args.gpus).split(","))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def txt_to_list(split_path):
    filename = []
    with open(split_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            filename.append(line)
    return filename 


def get_dataset_dist(config, logger, rank, world_size):
    # Init data loader
    if config["data.name"] == 'coco':
        from warpback.coco_dataset import COCODataset
        # train dataset & dataloader
        train_dataset = COCODataset(
            data_root=config['data.training_set_path'],
            depth_root=config['data.training_depth_path'],
            height=config['data.img_h'],
            width=config['data.img_w'],
            training=True,
            ec_weight_dir=config['data.ec_weight_dir'],
            rand_trans=config['data.rand_trans'],
            trans_range=config['data.trans_range'],
            rank=rank,
            trans_sign=config.get("data.trans_sign", [1, -1]),
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=config["data.per_gpu_batch_size"],
            drop_last=True, 
            num_workers=0,
            sampler=train_sampler, 
            collate_fn=train_dataset.collate_fn,
        )

        # val dataset & dataloader          
        val_dataset = COCODataset(
            data_root=config['data.val_set_path'],
            depth_root=config['data.val_depth_path'],
            height=config['data.img_h'],
            width=config['data.img_w'],
            training=True,
            ec_weight_dir=config['data.ec_weight_dir'],
            rand_trans=config['data.rand_trans'],
            trans_range=config['data.trans_range'],
            rank=rank,
            trans_sign=config.get("data.trans_sign", [1, -1])
        )
        val_collate_fn = val_dataset.collate_fn
        val_dataset = torch.utils.data.Subset(val_dataset, indices=list(range(len(val_dataset)))[::5])
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_data_loader_dict = {}
        val_data_loader_dict["coco"] = DataLoader(
            val_dataset,
            batch_size=config["data.per_gpu_batch_size"],
            sampler=val_sampler,
            shuffle=False, 
            drop_last=True, 
            num_workers=0, 
            collate_fn=val_collate_fn,
        )
    else:
        raise NotImplementedError

    return train_data_loader, val_data_loader_dict


def train(rank, world_size, config):
    setup(rank, world_size)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Config logging and tb writer
    logger = None
    if rank == 0:
        import logging
        # logging to file and stdout
        config["log_file"] = os.path.join(config["local_workspace"], "training.log")
        logger = logging.getLogger("mine")
        file_handler = logging.FileHandler(config["log_file"])
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.handlers = [file_handler, stream_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("Training config: {}".format(config))

        # tensorboard summary_writer
        config["tb_writer"] = SummaryWriter(log_dir=config["local_workspace"])
    config["logger"] = logger

    # Init data loader
    train_data_loader, val_data_loader = get_dataset_dist(config, logger, rank, world_size)
    synthesis_task = SynthesisTask(rank, config=config, logger=logger)
    synthesis_task.train(train_data_loader, val_data_loader)

    cleanup()


def main():
    # Prepare workspace
    workspace = config["local_workspace"]
    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
    
    with open(os.path.join(workspace, "params.yaml"), "w") as f:
        print("Dumping extra config file...")
        yaml.dump(config, f)

    # Start training
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
