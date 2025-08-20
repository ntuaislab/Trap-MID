import logging
import numpy as np
import os
import random
import statistics
import time
import torch
from argparse import ArgumentParser
from kornia import augmentation
import json

import losses as L
import utils
from evaluation import get_knn_dist, calc_fid
from models.classifiers import *
from models.generators.resnet64 import ResNetGenerator
from utils import save_tensor_images

print(f'>> Original number of threads: {torch.get_num_threads()}')
thread_rate = 0.25 if torch.get_num_threads() > 16 else 1
print(f'>> Set threading rate to be {thread_rate}')
torch.set_num_threads(int(torch.get_num_threads() * thread_rate))
print(f'>> Current number of threads: {torch.get_num_threads()}')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def inversion(args, G, T, E, iden, itr, lr=2e-2, iter_times=1500, num_seeds=5):
    save_img_dir = os.path.join(args.save_dir, 'all_imgs')
    success_dir = os.path.join(args.save_dir, 'success_imgs')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    bs = iden.shape[0]
    iden = iden.view(-1).long().cuda()

    G.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)
    no = torch.zeros(bs)  # index for saving all success attack images

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, num_seeds))

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    for random_seed in range(num_seeds):
        tf = time.time()
        r_idx = random_seed

        set_random_seed(random_seed)

        z = utils.sample_z(
            bs, args.gen_dim_z, device, args.gen_distribution
        )
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(iter_times):

            fake = G(z, iden)
            if args.data_name == 'mnist':
                # Expand chennel from 1 to 3 to fit pretrained models
                fake = fake.expand(-1, 3, -1, -1)

            out1 = T(aug_list(fake))[-1]
            out2 = T(aug_list(fake))[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            if args.inv_loss_type == 'ce':
                inv_loss = L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
            elif args.inv_loss_type == 'margin':
                inv_loss = L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
            elif args.inv_loss_type == 'poincare':
                inv_loss = L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            inv_loss_val = inv_loss.item()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake_img = G(z, iden)
                    if args.data_name == 'mnist':
                        # Expand chennel from 1 to 3 to fit pretrained models
                        fake_img = fake_img.expand(-1, 3, -1, -1)
                        eval_prob = E(fake_img)[-1]
                    else:
                        eval_prob = E(augmentation.Resize((112, 112))(fake_img))[-1]
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                    print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, inv_loss_val, acc))

        with torch.no_grad():
            fake = G(z, iden)
            if args.data_name == 'mnist':
                # Expand chennel from 1 to 3 to fit pretrained models
                fake = fake.expand(-1, 3, -1, -1)
                eval_prob = E(fake)[-1]
            else:
                eval_prob = E(augmentation.Resize((112, 112))(fake))[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

            cnt, cnt5 = 0, 0
            for i in range(bs):
                gt = iden[i].item()
                sample = G(z, iden)[i]
                all_img_class_path = os.path.join(save_img_dir, str(gt))
                if not os.path.exists(all_img_class_path):
                    os.makedirs(all_img_class_path)
                save_tensor_images(sample.detach(),
                                   os.path.join(all_img_class_path, "attack_iden_{}_{}.png".format(gt, r_idx)))

                if eval_iden[i].item() == gt:
                    seed_acc[i, r_idx] = 1
                    cnt += 1
                    flag[i] = 1
                    best_img = G(z, iden)[i]
                    success_img_class_path = os.path.join(success_dir, str(gt))
                    if not os.path.exists(success_img_class_path):
                        os.makedirs(success_img_class_path)
                    save_tensor_images(best_img.detach(), os.path.join(success_img_class_path,
                                                                       "{}_attack_iden_{}_{}.png".format(itr, gt,
                                                                                                         int(no[i]))))
                    no[i] += 1
                _, top5_idx = torch.topk(eval_prob[i], 5)
                if gt in top5_idx:
                    cnt5 += 1

            interval = time.time() - tf
            print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
            res.append(cnt * 1.0 / bs)
            res5.append(cnt5 * 1.0 / bs)
            torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5


if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Stage-2: Image Reconstruction')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--inv_loss_type', type=str, default='margin', help='ce | margin | poincare')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iter_times', type=int, default=600)
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # path
    parser.add_argument('--save_dir', type=str, default='PLG_MI_Inversion')
    parser.add_argument('--path_G', type=str, default='')

    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--eval_file', type=str, default='')
    parser.add_argument('--private_data_domain', type=str, default='datasets/celeba_private_domain',
                        help='path to private dataset root directory. default: CelebA')
    parser.add_argument('--private_data_feats', type=str, default='celeba_private_feats',
                        help='path to private features directory. default: CelebA')

    parser.add_argument('--num_classes', '-nc', type=int, default=1000,
                        help='Number of classes in training data.  default: 1000')
    parser.add_argument('--data_name', type=str, help='celeba | ffhq | facescrub | mnist')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of recovered image per class.  default: 5')

    args = parser.parse_args()
    logger = get_logger()

    logger.info(args)
    logger.info("=> creating model ...")

    set_random_seed(42)

    # load Generator
    G = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        num_classes=1000, distribution=args.gen_distribution,
        out_channels=1 if args.data_name == 'mnist' else 3
    )
    gen_ckpt_path = args.path_G
    gen_ckpt = torch.load(gen_ckpt_path)['model']
    G.load_state_dict(gen_ckpt)
    G = G.cuda()

    # Load target model
    if args.model == "VGG16":
        T = VGG16(args.num_classes)
    elif args.model == 'IR152':
        T = IR152(args.num_classes)
    elif args.model == "FaceNet64":
        T = FaceNet64(args.num_classes)
    elif args.model == "IR18":
        T = IR18(args.num_classes)
    elif args.model == 'VGG16_BiDO':
        T = VGG16_BiDO(args.num_classes)
    elif args.model == 'IR18_BiDO':
        T = IR18_BiDO(args.num_classes)
    else:
        raise NotImplementedError(f'Mode {args.model} not implemented.')
    path_T = args.ckpt_file
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)
    # T = T.module # disable data parallel to ensure reproducibility
    logger.info(f"=> Model loaded from {path_T}")

    # Load evaluation model
    if args.data_name == 'mnist':
        E = VGG16(10)
    else:
        E = FaceNet(args.num_classes)
    E = torch.nn.DataParallel(E).cuda()
    path_E = args.eval_file
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    # E = E.module # disable data parallel to ensure reproducibility

    logger.info("=> Begin attacking ...")
    aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    if args.data_name == 'mnist':
        # attack 5 classes, 5 classes per batch
        recovered_classes = 5
        batch_size = 5
    else:
        # attack 300 classes, 60 classes per batch
        recovered_classes = 300
        batch_size = 60
    iter_time = recovered_classes // batch_size
    
    iden = torch.from_numpy(np.arange(batch_size))

    # evaluate on the first 300 identities only
    for idx in range(iter_time):
        print("--------------------- Attack batch [%s]------------------------------" % idx)
        # reconstructed private images
        acc, acc5, var, var5 = inversion(args, G, T, E, iden, itr=0, lr=args.lr, iter_times=args.iter_times,
                                            num_seeds=args.num_seeds)

        iden = iden + batch_size
        aver_acc += acc / iter_time
        aver_acc5 += acc5 / iter_time
        aver_var += var / iter_time
        aver_var5 += var5 / iter_time

    print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(aver_acc, aver_acc5, aver_var, aver_var5))

    print("=> Calculate the KNN Dist.")
    knn_dist = get_knn_dist(E, os.path.join(args.save_dir, 'all_imgs'), args.private_data_feats, args)
    print("KNN Dist %.2f" % knn_dist)

    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(args.save_dir, "success_imgs"),
                   private_img_path=args.private_data_domain,
                   batch_size=100)
    print("FID %.2f" % fid)

    records = {
        'acc': aver_acc,
        'acc5': aver_acc5,
        'var': aver_var,
        'var5': aver_var5,
        'knn_dist': knn_dist,
        'fid': fid
    }
    with open(os.path.join(args.save_dir, 'result.json'), 'w') as f:
        json.dump(records, f, indent=4)
