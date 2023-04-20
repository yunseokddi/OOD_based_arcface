import os
import argparse
import torch.cuda
import logging
import torch.nn as nn

from utils.utils_config import get_config
from torch import distributed
from utils.utils_distributed_sampler import setup_seed
from torch.utils.tensorboard import SummaryWriter
from utils.utils_logging import init_logging, AverageMeter
from dataset import get_dataloader
from backbones import get_model
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from losses import CombinedMarginLoss
from partial_fc_v2 import PartialFC_V2
from lr_scheduler import PolynomialLRWarmup
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone.train()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    module_partial_fc = PartialFC_V2(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.train().cuda()
    # TODO the params of partial fc must be last in the params list
    opt = torch.optim.SGD(
        params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0

    criterion = nn.CrossEntropyLoss().cuda()
    loss_am = AverageMeter()

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)

            loss = criterion(local_embeddings, local_labels)

            loss.backward()
            if global_step % cfg.gradient_acc == 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()
                opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
