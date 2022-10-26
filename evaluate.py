import os
import time
import argparse
import datetime
import numpy as np
import sys
from utils import psnr_error
import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt
import torch.nn.functional as F


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--dataset', type=str, help="anomaly detection dataset to train")
    parser.add_argument('--model', type=str, help="the backbone of anomaly detection benchmark")
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
    test_gt = GT_loader(config)
    gt = test_gt.__test_gt__()
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)

    # select the testing model
    model.load_state_dict(torch.load(os.path.join("weights", "best.pth"), map_location='cpu'))
    model.cuda()
    with torch.no_grad():
        auc = eval_one_epoch(config, model, gt)
    print("AUC:", auc)


def eval_one_epoch(config, model, gt):
    model.eval()
    video_folders = os.listdir(config.PATH.TEST_FOLDER)
    video_folders.sort()
    video_folders = [os.path.join(config.PATH.TEST_FOLDER, aa) for aa in video_folders]
    batch_time = AverageMeter()
    start = time.time()
    end = time.time()
    score_group = []
    print('Start testing......')
    time_total = 0
    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = test_dataset(config, folder)
            score = []
            for idx, (img, mask) in enumerate(dataset):
                img = torch.from_numpy(img).unsqueeze(0).cuda()
                torch.cuda.synchronize()
                time_start = time.time()
                fea, y_pre, y = model(img, eval=True)
                loss_test = psnr_error(y_pre, y).cpu().data.detach().numpy()
                score.append(float(loss_test))
                torch.cuda.synchronize()
                time_total = time_total + time.time() - time_start
                batch_time.update(time.time() - end)
                end = time.time()
            score_group.append(np.array(score))
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

    # score_gap
    normal_scores = np.array([])
    abnormal_scores = np.array([])
    normal_nums = 0
    abnormal_nums = 0
    for i in range(len(labels)):
        if labels[i].item() == 1:
            abnormal_nums = abnormal_nums + 1
            abnormal_scores = np.concatenate((abnormal_scores, np.array([scores[i]])), axis=0)
        else:
            normal_nums = normal_nums + 1
            normal_scores = np.concatenate((normal_scores, np.array([scores[i]])), axis=0)
    print(sum(normal_scores) / normal_nums)
    print(sum(abnormal_scores) / abnormal_nums)

    print("推理时间效率：")
    print(time_total / (abnormal_nums + normal_nums))

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
    # seed = config.SEED + dist.get_rank()
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
    print(config.DATA.DATASET)
    main(config)
