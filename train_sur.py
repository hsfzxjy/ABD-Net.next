from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer
from torchreid.regularizers import get_regularizer


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
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu, args=vars(args), sur_dim=8)
    print(model)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion = get_criterion(dm.num_train_pids, use_gpu, args)
    regularizer = get_regularizer(vars(args))
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

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
            queryloader = testloader_dict[name]['query'], testloader_dict[name]['query_flip']
            galleryloader = testloader_dict[name]['gallery'], testloader_dict[name]['gallery_flip']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("==> Start training")

    if args.fixbase_epoch > 0:
        oldenv = os.environ.get('sa', '')
        os.environ['sa'] = ''
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=True)
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
        os.environ['sa'] = oldenv

    max_r1 = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        print(epoch)
        print(criterion)

        train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=False)
        train_time += round(time.time() - start_train_time)

        if use_gpu:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': 0,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query'], testloader_dict[name]['query_flip']
                galleryloader = testloader_dict[name]['gallery'], testloader_dict[name]['gallery_flip']
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if max_r1 < rank1:
                print('Save!', max_r1, rank1)
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_best.pth.tar'))

                max_r1 = rank1

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()


def train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=False):

    if not fixbase and args.use_of and epoch >= args.of_start_epoch:
        print('Using OF')

    from torchreid.losses.of_penalty import OFPenalty

    of_penalty = OFPenalty(vars(args))

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = None  # [AverageMeter() for _ in range(3)]

    model.train()

    if fixbase or args.fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, _, sur) in enumerate(trainloader):
        try:
            limited = float(os.environ.get('limited', None))
        except (ValueError, TypeError):
            limited = 1

        if not fixbase and (batch_idx + 1) > limited * len(trainloader):
            break

        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            sur = sur.cuda()

        outputs = model(imgs, sur)
        loss = criterion(outputs, pids)
        if not fixbase:
            reg = regularizer(model)
            loss += reg
        if not fixbase and args.use_of and epoch >= args.of_start_epoch:

            penalty = of_penalty(outputs)
            loss += penalty

        if acc is None:
            acc = [AverageMeter() for _ in range(len(outputs[1]))]

        for acc_meter, xent_feat in zip(acc, outputs[1]):
            acc_meter.update(accuracy(xent_feat, pids.cuda())[0].item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t{acc}'.format(
                      epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, acc=' '.join('{:.4f}'.format(x.avg) for x in acc)))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):

    flip_eval = args.flip_eval

    if flip_eval:
        print('# Using Flip Eval')

    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, q_paths = [], [], [], []

        if flip_eval:
            enumerator = enumerate(zip(queryloader[0], queryloader[1]))
        else:
            enumerator = enumerate(queryloader[0])

        for batch_idx, package in enumerator:
            end = time.time()

            if flip_eval:
                (imgs0, pids, camids, paths, sur0), (imgs1, _, _, _, sur1) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                    sur0, sur1 = sur0.cuda(), sur1.cuda()
                features = (model(imgs0, sur0)[0] + model(imgs1, sur1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, _, sur) = package
                if use_gpu:
                    imgs = imgs.cuda()
                    sur = sur.cuda()

                features = model(imgs, sur)[0]

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, g_paths = [], [], [], []
        if flip_eval:
            enumerator = enumerate(zip(galleryloader[0], galleryloader[1]))
        else:
            enumerator = enumerate(galleryloader[0])

        for batch_idx, package in enumerator:
            # print('fuck')
            end = time.time()

            if flip_eval:
                (imgs0, pids, camids, paths, sur0), (imgs1, _, _, _, sur1) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                    sur0, sur1 = sur0.cuda(), sur1.cuda()
                features = (model(imgs0, sur0)[0] + model(imgs1, sur1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, _, sur) = package
                if use_gpu:
                    imgs = imgs.cuda()
                    sur = sur.cuda()

                features = model(imgs, sur)[0]

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        if os.environ.get('save_feat'):
            import scipy.io as io
            io.savemat(os.environ.get('save_feat'), {'q': qf.data.numpy(), 'g': gf.data.numpy(), 'qt': q_pids, 'gt': g_pids})
            # return

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if os.environ.get('distmat'):
        import scipy.io as io
        io.savemat(os.environ.get('distmat'), {'distmat': distmat, 'qp': q_paths, 'gp': g_paths})

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
