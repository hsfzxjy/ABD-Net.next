from __future__ import print_function
from __future__ import division

import os
import sys
import time
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, image_dataset_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.utils.iotools import check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate


import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'CRITICAL'))

# global variables
parser = argument_parser()
args = parser.parse_args()

os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.torch'))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k.
    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.
    Returns:
        list: accuracy at top-k.
    Examples::
        >>> from torchreid import metrics
        >>> metrics.accuracy(output, target)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def get_criterion(num_classes: int, use_gpu: bool, args):

    if args.criterion == 'htri':

        from torchreid.losses.hard_mine_triplet_loss import TripletLoss
        criterion = TripletLoss(num_classes, vars(args), use_gpu)

    elif args.criterion == 'xent':

        from torchreid.losses.cross_entropy_loss import CrossEntropyLoss
        criterion = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    else:
        raise RuntimeError('Unknown criterion {}'.format(args.criterion))

    return criterion


def main():
    global args

    if not args.evaluate:
        raise RuntimeError('Test only!')

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stderr = sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu, args=vars(args))
    print(model)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        try:
            checkpoint = torch.load(args.load_weights)
        except Exception as e:
            print(e)
            checkpoint = torch.load(args.load_weights, map_location={'cuda:0': 'cpu'})

        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        state = model.state_dict()
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state)
        # args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            distmat = test(model, testloader_dict[name], use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

def test(model, loaders, use_gpu, ranks=[1, 5, 10, 20]):

    batch_time = AverageMeter()

    model.eval()

    def eval_set(name, loader):

        with torch.no_grad():
            qf, q_pids, q_camids, q_paths = [], [], [], []

            enumerator = enumerate(loader)

            for batch_idx, package in enumerator:
                end = time.time()

                (imgs, pids, camids, paths) = package
                if use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs)[0]

                batch_time.update(time.time() - end)

                features = features.data.cpu()
                qf.append(features)
                q_pids.extend(pids)
                q_camids.extend(camids)
                q_paths.extend(paths)
            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)

            print("Extracted features for {} set, obtained {}-by-{} matrix".format(name, qf.size(0), qf.size(1)))

            return qf, q_pids, q_camids

    results = {}
    for name, (qd, gd) in loaders.items():
        results[name] = (eval_set('{name} query'.format(name), qd), eval_set('{name} gallery'.format(name), gd))

    def eval_result(gr, qr):

        gf, g_pids, g_camids = gr
        qf, q_pids, q_camids = qr

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

        print("Results ----------")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
        print("------------------")

        return [mAP, *cmc]

    metrics = []
    for name, (qr, gr) in results.items():
        metrics.append(eval_result(gr, qr))
    metrics = np.average(np.array(metrics), axis=1)
    mAP, *cmc = metrics

    print('====> Average Result')
    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    return


if __name__ == '__main__':
    main()
