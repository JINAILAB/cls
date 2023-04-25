import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import transforms_cls
import argparse
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
import presets
import models
import utils
from torch import nn
import tqdm
import random
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import confusion_matrix

from torchvision.models import swin_v2_s
from torchvision.models import Swin_V2_S_Weights
from torchvision.models import efficientnet_v2_s
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models import regnet_y_16gf
from torchvision.models.regnet import RegNet_Y_16GF_Weights
from torchvision.models import swin_v2_t
from torchvision.models import Swin_V2_T_Weights

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="./hyfinal", type=str,
                        help="dataset path")

    parser.add_argument("--model", default="effnetv2_s", type=str, help="model name, resnet18, regnet_16gf, effnetv2_s, swinv2_m ")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--gpu", default=[0,1],  type=list)
    parser.add_argument("--distributed", default=True, type=bool )
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=21, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=2e-05,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=0.01,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.2, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0, type=float, help="mixup alpha (recommend: 0.2)")
    parser.add_argument("--cutmix-alpha", default=0, type=float, help="cutmix alpha (recommend: 1.0)")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int,
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./model_pth", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--valid-only",
        dest="valid_only",
        help="Only valid the model",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--test", default=False, type=bool, help='including test_')
    # distributed training parameters
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=300, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=280, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=280, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--channel1", default=False, type=bool, help='use only 1channel'
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float,
                        help="the maximum gradient norm (default None)")
    parser.add_argument("--random-erase", default=0.0, type=float,
                        help="random erase recommend 0.1")
    parser.add_argument("--auto-augment", default=None, type=str,
                        help="ra, ta_wide, augmix")
    

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--use_deterministic_algorithms", default=False, type=bool)
    parser.add_argument("--confusion_matrix", default = False, type=bool)
    return parser



def train_one_epoch(net, criterion, optimizer, data_loader, device, epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        current_correct = predicted.eq(targets).sum().item()
        correct += current_correct

    # train_acc.append(correct / total)

    print('\nTotal average train accuarcy:', correct / total)
    print('Total average train loss:', train_loss / total)


def evaluate(net, criterion,  valid_loader, device, epoch, args):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    loss = 0
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss += criterion(outputs, targets).item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            
            if args.confusion_matrix:
            
                output = (torch.max(outputs, 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction
            
                labels = targets.data.cpu().numpy()
                y_true.extend(labels) # Save Truth

        print('\nTotal average test accuracy:', correct / total)
        print('Total average test loss:', loss / total)
        
        if args.confusion_matrix:
            cf_matrix = confusion_matrix(y_true, y_pred)
            print(cf_matrix)
            return (correct / total) , cf_matrix
                
    # Build confusion matrix



    return correct / total


def load_data(traindir, valdir, args):
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)
    

    print("Loading training data")
    st = time.time()

    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    channel1 = args.channel1
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            channel1 = channel1,
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        preprocessing = weights.transforms()
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation, channel1=channel1
        )

    dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )


    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    device = torch.device("cuda")
    print('model is', args.model)

    interpolation = InterpolationMode(args.interpolation)

    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    


    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms_cls.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms_cls.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")


    if args.model == 'resnet18':
        model = models.EModel(resnet18(weights = ResNet18_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'regnet_16gf':
        model = models.EModel(regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2), len(dataset.classes))
    elif args.model == 'effnetv2_s':
        model = models.EModel(efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'mlp':
        model = models.MLP(len(dataset.classes))
    elif args.model == 'swinv2_t':
        model = models.EModel(swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'swinv2_s':
        model = models.EModel(swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1), len(dataset.classes))

    #model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        # if not args.test_only:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        evaluate(model, criterion, data_loader_test, device=device, args=args)
        return

    print("Start training")
    start_time = time.time()
    max_eval_acc = 0
    max_matrix = []
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        if args.confusion_matrix:
            eval_acc, c_matrix = evaluate(model, criterion, data_loader_test, epoch=epoch, device=device, args=args)
        else:
            eval_acc = evaluate(model, criterion, data_loader_test, epoch=epoch, device=device, args=args)
        

        if max_eval_acc < eval_acc:
            max_eval_acc = eval_acc
            if args.confusion_matrix:
                max_matrix = c_matrix
            torch.save(model.state_dict(), os.path.join(args.output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_best.pth"))
    
    print('dataset class는', dataset.classes)
    torch.save(model.state_dict(), os.path.join(args.output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_best.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print(args.model, 'max_acc is ', max_eval_acc)
    with open('./train_log/' + str(args.data_path).split('/')[-1] +'_'+args.model+".txt", 'a') as f:
        f.write('dataset class는' + str(dataset.classes) + '\n')
        f.write(args.model + ' max_acc is ' + str(max_eval_acc)+ '\n')
        if args.confusion_matrix:
            f.write(str(c_matrix)+'\n')

    if args.test:
        import cv2
        import glob
        from torchvision import transforms
        from torch.utils.data import Dataset, DataLoader
        class Testset(Dataset):
            def __init__(self, image_folder, transforms):        
                self.image_folder = image_folder   
                self.transforms = transforms

            def __len__(self):
                return len(self.image_folder)
            
            def __getitem__(self, index):        
                image_fn = self.image_folder[index]                                       
                image = cv2.imread(image_fn, cv2.IMREAD_COLOR)        
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transforms:            
                    image = self.transforms(image)

                return image
        
        test_list = glob.glob(args.data_path+'/test/*.png')
        test_list=sorted(test_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        test_transform = transforms.Compose([
            transforms.ToTensor(), #이미지 데이터를 tensor 데이터 포멧으로 바꾸어줍니다.
            transforms.Resize([300,300]), #이미지의 크기가 다를 수 있으니 크기를 통일해 줍니다.
            #transforms.Normalize(mean=(0.139, 0, 0), std=(0.073, 1, 1)) #픽셀 단위 데이터를 정규화 시켜줍니다.
        ])
        test_data = Testset(test_list, test_transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        
        model.eval()
        test_idx = []
        for x in test_loader:
            x = x.to(device)
            outputs = model(x)
            predicted = outputs.argmax(1)
            predicted = predicted.detach().cpu().numpy().tolist()
            test_idx.extend(predicted)
            
        print(test_idx)
        with open('./train_log/' + str(args.data_path).split('/')[-1] +'_'+args.model+".txt", 'a') as f:
            f.write(str(test_list)+'\n')
            f.write(str(test_idx))
            
    
    























if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)