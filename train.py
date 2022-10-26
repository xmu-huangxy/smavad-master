import os
import time
import argparse
import datetime
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from utils import psnr_error, compute_kl_loss
import torch
import torch.backends.cudnn as cudnn
from timm.utils import AverageMeter
from torch.utils import data
from config import get_config
from models import build_model
from data.Dataset import DataLoader, GT_loader, test_dataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from sklearn.metrics import roc_curve
from sklearn import metrics
from tqdm import tqdm


try:
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('SMA-VAD training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--dataset', type=str, help="anomaly detection dataset to train")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    train_dataset = DataLoader(config, train=True)
    data_loader_train = data.DataLoader(train_dataset, config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, shuffle=True)
    test_gt = GT_loader(config)
    gt = test_gt.__test_gt__()
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)

    # initial the parameters of model, you can download the .pth and put it in check_point directory
    # the download link is https://drive.google.com/file/d/1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw/view
    checkpoint = torch.load('./check_point/simmim_pretrain__vit_base__img224__800ep.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    logger.info("Start training")
    start_time = time.time()
    best_score = -1
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        with torch.no_grad():
            auc = eval_one_epoch(config, model, gt)
            if auc > best_score:
                best_score = auc
                torch.save(model.state_dict(), f'weights/best_avenue.pth')
    print("Max_AUC:", best_score)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask) in tqdm(enumerate(data_loader), desc="Training Epoch %d" % (epoch + 1), total=len(data_loader)):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        loss1, fea1 = model(img, mask, is_mask=True)
        loss2, fea2 = model(img, mask)
        fea_loss = compute_kl_loss(fea1, fea2)
        loss = loss1 + loss2 + 0.3 * fea_loss

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def eval_one_epoch(config, model, gt):
    model.eval()
    video_folders = os.listdir(config.PATH.TEST_FOLDER)
    video_folders.sort()
    video_folders = [os.path.join(config.PATH.TEST_FOLDER, aa) for aa in video_folders]
    batch_time = AverageMeter()
    start = time.time()
    end = time.time()
    score_group = []
    fea_group = []
    print('Start testing......')
    for i, folder in enumerate(video_folders):
        dataset = test_dataset(config, folder)
        score = []
        fea_score = []
        for idx, (img, mask) in enumerate(dataset):
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            fea, img_rec, img_target = model(img, eval=True)
            test_psnr = psnr_error(img_target, img_rec).cpu().detach().numpy()
            score.append(float(test_psnr))
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()
        score_group.append(np.array(score))
        fea_group.append(np.array(fea_score))
    epoch_time = time.time() - start
    labels = np.array(gt)
    scores = np.array([], dtype=np.float32)
    if config.DATA.DATASET == "avenue":
        distance = np.array([], dtype=np.float32)
        for i in range(len(score_group)):
            distance = np.concatenate((distance, score_group[i]), axis=0)
        distance -= min(distance)
        distance /= max(distance)
        scores = np.concatenate((scores, 1 - distance), axis=0)
    else:
        for i in range(len(score_group)):
            distance = score_group[i]
            distance -= min(distance)
            distance /= max(distance)
            scores = np.concatenate((scores, 1 - distance), axis=0)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC: {auc}\n')
    logger.info(f"testing takes {datetime.timedelta(seconds=int(epoch_time))}")
    return auc


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    # print config
    logger.info(config.dump())
    main(config)
