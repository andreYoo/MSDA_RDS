import pdb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSet,ImgDataSet_TARGET
import os
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from glob import glob
from itertools import cycle
from utils import Results_Visualiastion, Results_Visualiastion_v2
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
writer = SummaryWriter('./logs2')
DNAME = ['POTHOLE','CRACK','TARGET']
TARGET_DOMAIN_TRAIN = 10

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device, type ='vgg16'):
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    else:
        assert False
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizers, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    _len = len(optimizers)
    for i in range(_len):
        lr = lr * 0.95
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = lr
    writer.add_scalar("Learning_rate", lr, epoch)

def find_latest_model_path(_dir,num_model):
    model_paths = []
    epochs = []
    rt_list = []
    for _i in range(num_model):
        dir = os.path.join(_dir,'model-%d'%(_i))
        for path in Path(dir).glob('*.pt'):
            if 'epoch' not in path.stem:
                continue
            model_paths.append(path)
            parts = path.stem.split('_')
            epoch = int(parts[-1])
            epochs.append(epoch)

        if len(epochs) > 0:
            epochs = np.array(epochs)
            max_idx = np.argmax(epochs)
            rt_list.append(model_paths[max_idx])
        else:
            return None
        return rt_list

def validate(models, val_loaders, criterion):
    losses = []
    _max_train_loader = 0
    _max_tl_index = -1
    for _s, _tload in enumerate(val_loaders):
        if _tload.__len__() > _max_train_loader:
            _max_train_loader = _tload.__len__()
            _max_tl_index = _s
    for idx, model in enumerate(models):
        model.eval()
        losses.append(AverageMeter())
    if _max_tl_index == 0:
        _dataloader = zip(val_loaders[0], cycle(val_loaders[1]))
    else:
        _dataloader = zip(cycle(val_loaders[0]), val_loaders[1])
    with torch.no_grad():
        for i, inputs in enumerate(_dataloader):
            for _idx, _input in enumerate(inputs):
                input, target = _input
                input_var = Variable(input).cuda()
                target_var = Variable(target).cuda()
                output = models[_idx](input_var)
                loss = criterion(output, target_var)
                losses[idx].update(loss.item(), input_var.size(0))
    return {'valid_loss_1': losses[0].avg, 'valid_loss_2': losses[1].avg}


def validate_with_visualisation(models, val_loaders, criterion,epoch,if_target=False):
    losses = []
    _max_train_loader = 0
    _max_tl_index = -1
    for _s,_tload in enumerate(val_loaders):
        if _tload.__len__() > _max_train_loader:
            _max_train_loader = _tload.__len__()
            _max_tl_index = _s
    for idx,model in enumerate(models):
        model.eval()
        losses.append(AverageMeter())
    if _max_tl_index==0:
        _dataloader = zip(val_loaders[0],cycle(val_loaders[1]))
    else:
        _dataloader = zip(cycle(val_loaders[0]),val_loaders[1])
    with torch.no_grad():
        for i, inputs in enumerate(_dataloader):
            for _idx, _input in enumerate(inputs):
                input,target = _input
                input_var = Variable(input).cuda()
                target_var = Variable(target).cuda()
                output,_latent = models[_idx](input_var,if_latent=True)
                mask = F.sigmoid(output).data.cpu().numpy()
                loss = criterion(output, target_var)
                losses[idx].update(loss.item(), input_var.size(0))
                if i%10==0:
                    if if_target==False:
                        Results_Visualiastion(input_var,target_var,output,mask,0.1,DNAME[_idx],i,epoch)
                        _tfname = './distribution/%s-epoch-%d-step-%d.png' % (DNAME[_idx],epoch, i)
                        latent_distribution(_tfname, _latent, target_var)
                    else:
                        Results_Visualiastion(input_var,target_var,output,mask,0.1,DNAME[_idx],i,epoch)
                        _tfname = './distribution/target_%s-epoch-%d-step-%d.png' % (DNAME[_idx],epoch, i)
                        latent_distribution(_tfname, _latent, target_var)
    return {'valid_loss_1': losses[0].avg,'valid_loss_2': losses[1].avg}


def visualisation(model,model_index, val_loader,epoch,domain_idx,if_target=False):
    _max_train_loader = 0
    _max_tl_index = -1
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            if if_target==True:
                input = inputs
                input_var = Variable(input).cuda()
            else:
                input,target = inputs
                input_var = Variable(input).cuda()
                target_var = Variable(target).cuda()
            output,_latent = model(input_var,if_latent=True)
            mask = F.sigmoid(output).data.cpu().numpy()
            if i%10==0:
                if if_target==False:
                    Results_Visualiastion(input_var,target_var,output,mask,0.1,DNAME[domain_idx],i,epoch)
                    _tfname = './distribution/%s-model-%d-epoch-%d-step-%d.png' % (DNAME[domain_idx],model_index,epoch, i)
                    latent_distribution(_tfname, _latent, target_var)
                else:
                    Results_Visualiastion(input_var,output,output,mask,0.1,DNAME[domain_idx],i,epoch)
                    _tfname = './distribution/target_%s-model-%d-epoch-%d-step-%d.png' % (DNAME[domain_idx],model_index,epoch, i)
                    #latent_distribution(_tfname, _latent, target_var)


def latent_distribution(ttfilename,_latent,label):
    _latent = _latent.detach().cpu().numpy()
    _label  = label.detach().cpu().numpy()
    _len =len(_latent)
    for i in range(_len):
        tmp = np.transpose(_latent[i],(1,2,0))
        tmp_label = np.transpose(_label[i],(1,2,0))
        _shape = np.shape(tmp)
        _tlatent = tmp.reshape(_shape[0]*_shape[1],_shape[2])
        _tlabel = tmp_label.reshape(_shape[0]*_shape[1])
        #vis_embedding = TSNE(n_components=2,random_state=0,perplexity=50).fit_transform(_tlatent)
        vis_embedding = PCA(n_components=2).fit_transform(_tlatent)
        fig = plt.figure()
        _filename = '.'+ttfilename.split('.')[1]+'-img-%d'%(i)+ttfilename.split('.')[2]
        plt.scatter(vis_embedding[_tlabel==0,0],vis_embedding[_tlabel==0,1],color='blue',s=0.5,alpha=0.7)
        plt.scatter(vis_embedding[_tlabel==1,0],vis_embedding[_tlabel==1,1],color='red',s=0.5,alpha=0.7)
        plt.savefig(_filename)
        plt.close()



def train(train_loader_list, models, criterion, optimizers,schedulers, valid_loader_list, args):
    train_loaders = train_loader_list[0]
    target_train_loader = train_loader_list[1]
    latest_model_paths = find_latest_model_path(args.model_dir,len(models))
    best_model_paths = []
    for i in range(len(models)):
        best_model_paths.append(os.path.join(*[args.model_dir,'model-%d'%(i), 'model-%d_best.pt'%(i)])) #Road pre-train model

    valid_loaders = valid_loader_list[0]
    target_valid_loader = valid_loader_list[1]

    epoch = 0
    min_val_loss = [99999,99999,99999]
    if latest_model_paths!=None:
        for i,latest_model_path in enumerate(latest_model_paths):
            if latest_model_path is not None:
                state = torch.load(latest_model_path)
                models[i].load_state_dict(state['model'])
                if i==0:
                    epoch = state['epoch']
                    epoch = epoch+1
                    print(f'Started training model from epoch {epoch}')

                #if latest model path does exist, best_model_path should exists as well
                assert Path(best_model_paths[i]).exists() == True, f'best model path {best_model_paths[i]} does not exist'
                #load the min loss so far
                best_state = torch.load(latest_model_path)
                min_val_los = best_state['valid_loss']
                print(f'Restored {i}-th model. Min validation loss so far is : {min_val_los}')
            else:
                print('Started training model from epoch 0')
                epoch = 0
                min_val_los = 9999
                break
    print('Learning rate renewal')
    if epoch!=0:
        for i in range(epoch):
            for _s in schedulers:
                _s.step()

    _max_train_loader = 0
    _max_tl_index = -1
    for _s,_tload in enumerate(train_loaders):
        if _tload.__len__() > _max_train_loader:
            _max_train_loader = _tload.__len__()
            _max_tl_index = _s
    _s_step = 0
    for epoch in range(epoch, args.n_epoch + 1):
        tq = tqdm.tqdm(total=(len(train_loaders[_max_tl_index]) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')
        losses = []
        t_losses = []
        for _model in models:
            _model.train()
            losses.append(AverageMeter())
            t_losses.append(AverageMeter())
        for i, inputs in enumerate(zip(cycle(train_loaders[0]),train_loaders[1],cycle(target_train_loader))):
            l_list = []
            for opt in optimizers:
                opt.zero_grad()
            loss = []
            for _idx, _input in enumerate(inputs[0:2]):
                input,target = _input
                input_var  = Variable(input).cuda()
                target_var = Variable(target).cuda()
                masks_pred,latent = models[_idx](input_var,if_latent=True)
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat  = target_var.view(-1)
                l_list.append(latent)
                _loss = criterion[0](masks_probs_flat, true_masks_flat)
                loss.append(_loss)
                writer.add_scalar("Model-%d-train/Source_domain_Loss"%(_idx), _loss, _s_step)
                losses[_idx].update(_loss)
                #tq.set_postfix(loss='{:.5f}'.format(losses.avg))
                #tq.update(args.batch_size)
                # compute gradient and do SGD step

            # Target Segmentation learnign and domain adapataion
            if epoch<TARGET_DOMAIN_TRAIN:
                t_input = inputs[2] #Target Input
                t_input_var  = Variable(t_input).cuda()
                _shape = np.shape(t_input_var)
                pred_list = []
                t_l_list = []
                hard_pseudo_label = torch.zeros((_shape[0],1,_shape[2],_shape[3])).cuda()
                soft_pseudo_label = -1.0*torch.ones(_shape[0],1,_shape[2],_shape[3]).cuda()
                target_loss = []
                for _i in range(len(models)):
                    t_masks_pred,t_latent = models[_i](t_input_var, if_latent=True)
                    pred_list.append(F.sigmoid(t_masks_pred))
                    t_l_list.append(t_latent)
                hard_pseudo_label[torch.mean(torch.stack(pred_list),dim=0)>0.15]=1.0
                soft_pseudo_label[torch.mean(torch.stack(pred_list),dim=0)>0.1] = 1.0
                soft_pseudo_label[torch.mean(torch.stack(pred_list),dim=0)<0.01] = 0.0
                for _i in range(len(models)):
                    t_hard_seg_loss = torch.mean(criterion[1](pred_list[_i].view(-1), hard_pseudo_label.view(-1)))
                    r_latent = t_l_list[_i].flatten(2,3).transpose(1,2)[torch.squeeze(soft_pseudo_label).flatten(1,2)==0.0,:]
                    r_c = torch.mean(r_latent,dim=0)
                    c_latent = t_l_list[_i].flatten(2,3).transpose(1,2)[torch.squeeze(soft_pseudo_label).flatten(1,2)==1.0,:]

                    _road = torch.mean((r_latent-r_c)**2)
                    _crack = torch.mean((c_latent-r_c)**-2.0)
                    t_soft_seg_loss= 0.1*(_road+_crack)
                    if torch.isnan(t_soft_seg_loss) or torch.isinf(t_soft_seg_loss):
                        t_soft_seg_loss = 0.0
                    _ttarget_loss = t_hard_seg_loss+t_soft_seg_loss
                    writer.add_scalar("Model-%d-train/Target_domain_Loss" % (_i), _ttarget_loss, _s_step)
                    target_loss.append(t_hard_seg_loss+t_soft_seg_loss)
                    t_losses[_i].update(t_soft_seg_loss)
                for _loss,_target_loss in zip(loss,target_loss):
                    total_loss = _loss+_target_loss
                    total_loss.backward()
            else:
                for _loss in loss:
                    _loss.backward()
            #Train for target_domain_and domain adpatation
            #Segmentation loss for target domain
            for opt in optimizers:
                opt.step()
            tq.set_postfix(
                loss='Model 1 {:.5f} Model 2 {:.5f}'.format(losses[0].avg, losses[1].avg))

            #tq.set_postfix(loss='Model 1 {:.5f} Model 2 {:.5f} Model E {:.5f}'.format(losses[0].avg,losses[1].avg,losses[2].avg))
            tq.update(args.batch_size)
            writer.add_scalar("Learning_rate", schedulers[0].get_last_lr()[0], _s_step)
            _s_step +=1
        valid_metrics = validate(models, valid_loaders, criterion[0])
        valid_loss = []
        valid_loss.append(valid_metrics['valid_loss_1'])
        valid_loss.append(valid_metrics['valid_loss_2'])
        if epoch%5==0:
            visualisation(models[0],0, valid_loaders[0],epoch,0)
            visualisation(models[1],0, valid_loaders[1],epoch,1)
            visualisation(models[0],0, target_train_loader,epoch,-1,if_target=True)
            visualisation(models[1],0, target_train_loader,epoch,-1,if_target=True)
        for _j,_v_loss in enumerate(valid_loss):
            writer.add_scalar("Model-%d-train/Valid_Loss"%(_j), _v_loss, epoch)
        print(f'\tvalid_loss_1 = {valid_loss[0]:.5f} | valid_loss_2 {valid_loss[1]:.5f}')
        tq.close()

        #save the model of the current epoch
        for _m_idx,_model in enumerate(models):
            epoch_model_path = os.path.join(*[args.model_dir,'model-%d'%(_m_idx), f'model-{_m_idx}_epoch_{epoch}.pt'])
            torch.save({
                'model': _model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss[_m_idx],
                'train_loss': losses[_m_idx].avg,
            }, epoch_model_path)
            print('[SAVED] %s'%(epoch_model_path))
            if valid_loss[_m_idx] < min_val_loss[_m_idx]:
                min_val_loss[_m_idx] = valid_loss[_m_idx]
                torch.save({
                    'model': _model.state_dict(),
                    'epoch': epoch,
                    'valid_loss': valid_loss[_m_idx],
                    'train_loss': losses[_m_idx].avg,
                }, best_model_paths[_m_idx])
                print('[Renewal Best Model] %s'%(best_model_paths[_m_idx]))
            schedulers[_m_idx].step()

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=4, type=int,  help='Batch size (defult: 4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')

    parser.add_argument('-source_data_dirs', default='./train_db',type=str, help='input source dataset (labelled) directory')
    parser.add_argument('-target_data_dir',default='./targets', type=str, help='input target dataset (unlabelled (image only)) directory')
    parser.add_argument('-model_dir',default='./models4', type=str, help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='resnet34', choices=['vgg16', 'resnet101', 'resnet34'])

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    DIR_IMG = []
    DIR_MASK = []
    Source_paths =  glob(os.path.join(args.source_data_dirs,"*/"))

    for _sp in Source_paths:
        DIR_IMG.append(os.path.join(_sp, 'images'))
        DIR_MASK.append(os.path.join(_sp, 'masks'))

    TARGET_DIR_IMG = os.path.join(args.target_data_dir, 'images')
    TARGET_DIR_MASK = os.path.join(args.target_data_dir, 'masks')

    img_names = []
    mask_names = []
    for _i in range(len(DIR_IMG)):
        if _i==0:
            img_names.append([path.name for path in Path(DIR_IMG[_i]).glob('*.png')])
            mask_names.append([path.name for path in Path(DIR_MASK[_i]).glob('*.png')])
        else:
            img_names.append([path.name for path in Path(DIR_IMG[_i]).glob('*.jpg')])
            mask_names.append([path.name for path in Path(DIR_MASK[_i]).glob('*.jpg')])

    MODEL_NUM = len(DIR_IMG)

    target_img_names = [path.name for path in Path(TARGET_DIR_IMG).glob('*.png')]
    target_mask_names = [path.name for path in Path(TARGET_DIR_MASK).glob('*.png')]

    print(f'The number of source datasets  ={len(img_names)}')
    for i in range(len(img_names)):
        print(f'total images of {i}-source dataset = {len(img_names[i])}')
    print(f'total images of target dataset = {len(target_img_names)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = []
    for i in range(MODEL_NUM):
        models.append(create_model(device, args.model_type))

    optimizers = []
    schedulers = []
    for i in range(MODEL_NUM):
        optimizers.append(torch.optim.SGD(models[i].parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay))
        schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[i], gamma=0.95))

    #crack_weight = 0.4*calc_crack_pixel_weight(DIR_MASK)
    #print(f'positive weight: {crack_weight}')
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([crack_weight]).to('cuda'))
    criterion1 = nn.BCEWithLogitsLoss().to('cuda')
    criterion2 = nn.BCEWithLogitsLoss(reduction='none').to('cuda')

    criterion = [criterion1,criterion2]


    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((448,448)),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((448,448)),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))])

    #Dataload for source dataset=====================================================================
    datasets = []
    train_datasets = []
    valid_datasets = []
    train_loaders = []
    valid_loaders = []
    for i in range(len(DIR_IMG)):
        dataset = ImgDataSet(img_dir=DIR_IMG[i], img_fnames=img_names[i], img_transform=train_tfms, mask_dir=DIR_MASK[i], mask_fnames=mask_names[i], mask_transform=mask_tfms)
        datasets.append(dataset)
        train_size = int(0.98*len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)


    # Dataload for target dataset=====================================================================
    target_dataset = ImgDataSet_TARGET(img_dir=TARGET_DIR_IMG, img_fnames=target_img_names, img_transform=train_tfms)
    target_train_size = int(0.98 * len(target_dataset))
    target_valid_size = len(target_dataset) - target_train_size
    target_train_dataset, target_valid_dataset = random_split(target_dataset, [target_train_size, target_valid_size])

    target_train_loader = DataLoader(target_train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                              num_workers=args.num_workers)
    target_valid_loader = DataLoader(target_valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(),
                              num_workers=args.num_workers)


    for i in range(MODEL_NUM):
        models[i].cuda()

    train([train_loaders,target_train_loader], models, criterion, optimizers,schedulers, [valid_loaders,target_valid_loader], args)

