# Author: Hirak J. Kashyap
import argparse
import matplotlib.pyplot as plt
import numpy as np
import refresh_transforms
from utils import flowlib, printlib, rotation_lib, evaluate_rpe_tum, motion_field, object_motion_lib
from models.mfg_1000mixed_sintel_model import MFG1000Mixed
from data.sintel_data import SintelDataset
import torch.backends.cudnn as cudnn
from models.PWC_SN import PWCNet # PWC net from https://github.com/sniklaus/pytorch-pwc
import loss_functions
import torch.optim
import log_tools
import csv
from tensorboardX import SummaryWriter
import torchvision
import sys

##########################################################

assert (int(torch.__version__.replace('.', '')) >= 40)  # requires at least pytorch version 0.4.0

# torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

cudnn.enabled = True  # make sure to use cudnn for computational performance

cudnn.benchmark = True  # checks for the optimal algorithm for a fixed input size

##########################################################

parser = argparse.ArgumentParser()
np.set_printoptions(threshold=sys.maxsize)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--data_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, choices=["virtual_kitti", "kitti_odom"], default="virtual_kitti")
parser.add_argument('--height', type=int, default=192, help="image height") #448
parser.add_argument('--width', type=int, default=512, help="image width") # 1024
parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--log_freq', default=100, type=int, help='Log after every x frame')
parser.add_argument('--log_im_freq', default=0, type=int, help='Log image outputs after every x frame')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--w_mft', type=float, default=1.0, help='Weight of translational MF reconstruction loss')
parser.add_argument('--w_mfw', type=float, default=1.0, help='Weight of rotational MF reconstruction loss')
parser.add_argument('--w_sparse', type=float, default=1.0, help='Weight of sparsity loss')
parser.add_argument('--stop', default=100000, type=int, help='Stop after number of data points')
parser.add_argument('--start', default=0, type=int, help='Start from frame number')
parser.add_argument("--shuffle", type=str2bool, nargs='?', const=True, default='n',
                    help="Shuffle input data.")
parser.add_argument("--train_on", type=str2bool, nargs='?', const=True, default='n',
                    help="Whether to train on Sintel dataset.")
parser.add_argument("--true_size", type=str2bool, nargs='?', const=True, default='n',
                    help="Do not resize inputs")
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--weight_decay', default=0., type=float, metavar='W', help='weight decay')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--pretrained_mfg', dest='pretrained_mfg', default=None, metavar='PATH',
                    help='path to pre-trained MFG model')
parser.add_argument("--opt_algo", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--plot_data", type=str2bool, nargs='?', const=True, default='n',
                    help="To plot input images, flow, and depth.")
parser.add_argument("--save_data", type=str2bool, nargs='?', const=True, default='n',
                    help="To save data to disk")

n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    args = parser.parse_args()
    scale_input = refresh_transforms.Rescale(args.height, args.width)

    train_drives = ['ambush_2',
                    'ambush_4',
                    'ambush_7',
                    'bamboo_1',
                    'bamboo_2',
                    'bandage_2',
                    'cave_2',
                    'market_2',
                    'market_6',
                    'mountain_1',
                    'shaman_3',
                    'sleeping_2',
                    'temple_3']

    val_drives = ['alley_1', 'ambush_5', 'bandage_1', 'shaman_2', 'sleeping_1']

    if args.shuffle:
        print('Shuffle data ON')

    global n_iter
    if args.train_on:
        print('Training on Sintel: ON')
    else:
        print('Training on Sintel: OFF')
    print('sparsity coeff:', args.w_sparse)
    print('learning rate:', args.lr)
    print('batch_size:', args.batch_size)
    print('momentum:', args.momentum)
    print('weight decay: ', args.weight_decay)
    print('train drives:', train_drives)
    print('validation drives:', val_drives)

    # load pretrained optic flow network PWCNet
    # pwc downscales by 4 times, change to 1.0 if flow GT is used instead
    args.downscale_f = 4.0
    flow_model = PWCNet().to(device)
    flow_parameters_file = 'pretrained_param/sintel_flow.pytorch'

    flow_parameters = torch.load(flow_parameters_file)
    if 'state_dict' in flow_parameters.keys():
        flow_model.load_state_dict(flow_parameters['state_dict'])
    else:
        # This is executed
        flow_model.load_state_dict(flow_parameters)

    flow_model = torch.nn.DataParallel(flow_model)
    flow_model.eval()

    # Initialize MFG model
    mf_model = MFG1000Mixed(round(args.height/args.downscale_f), round(args.width/args.downscale_f)).to(device)
    if args.pretrained_mfg:
        print(" ######### Using pretrained weights for MFG net #########")
        pretrained_model = torch.load(args.pretrained_mfg)
        mf_model.load_state_dict(pretrained_model['state_dict'])
        pretrained_epoch = pretrained_model['epoch']
    else:
        if not args.train_on:
            print('Pretrained network not provided for validation, exiting!')
            sys.exit()
        else:
            mf_model.init_weights()
            pretrained_epoch = 0

    args.epochs += pretrained_epoch

    mf_model = torch.nn.DataParallel(mf_model)
    if args.train_on:
        mf_model.train()
        optim_params = [
            {'params': mf_model.parameters(), 'lr': args.lr}
        ]

        if args.opt_algo == 'adam':
            optimizer = torch.optim.Adam(optim_params,
                                         betas=(args.momentum, args.beta),
                                         weight_decay=args.weight_decay)
            print('adam solver is set, now training ...')

        if args.opt_algo == 'sgd':
            optimizer = torch.optim.SGD(optim_params,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            print('SGD solver is set, now training ...')

        # create a handler object for the training dataset
        data_to_train = SintelDataset(dataset_dir=args.data_dir,
                                      drives=train_drives,
                                      transform=refresh_transforms.Compose([scale_input,
                                                                            refresh_transforms.ArrayToTensor()]))

        print('training data size: ', data_to_train.num_samples, ' frames')

        train_data_loader = torch.utils.data.DataLoader(data_to_train,
                                                        batch_size=args.batch_size,
                                                        shuffle=args.shuffle,
                                                        num_workers=args.workers,
                                                        pin_memory=True,
                                                        drop_last=True)
    else:
        mf_model.eval()
        torch.set_grad_enabled(False)

    print('MF-model parameters: ', sum(p.numel() for p in mf_model.parameters()))

    save_path = log_tools.save_path_formatter_sintel(args, parser)
    args.save_path = '/data/motion/smd/train_sintel'/ save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    training_writer = SummaryWriter(args.save_path)

    with open(args.save_path / 'test_RPE_summary.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['epoch', '-----drive-------', '-----trans error mean ------',
                         '-----rot error mean-----', '-----trans error std ------', '-----rot error std-----'])

    best_val_error = sys.maxsize
    for epoch in range(args.epochs - pretrained_epoch):
        cum_epoch = 0
        if args.train_on and len(train_drives) > 0:
            train_loss = train(train_data_loader, flow_model, mf_model, args, optimizer, training_writer)
            train_loss = train_loss / data_to_train.num_samples
            print('epoch ', epoch, ' --- training loss: ', train_loss)
            cum_epoch = epoch + 1 + pretrained_epoch
        else:
            cum_epoch = pretrained_epoch

        # create a handler object for the validation dataset
        n_active_list_alldrive = []
        for i_drive, drive in enumerate(val_drives):
            data_to_validate = SintelDataset(dataset_dir=args.data_dir,
                                             drives=[drive],
                                             transform=refresh_transforms.Compose([scale_input,
                                                                                   refresh_transforms.ArrayToTensor()]))

            val_data_loader = torch.utils.data.DataLoader(data_to_validate,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=args.workers,
                                                          pin_memory=True,
                                                          drop_last=False)

            terror_mean, terror_std, rerror_mean, rerror_std, epe, n_active_list_drive = validate(val_data_loader, flow_model, mf_model, args, drive, data_to_validate.num_samples)
            with open(args.save_path / 'test_RPE_summary.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(
                    [cum_epoch, drive, terror_mean, rerror_mean, terror_std,  rerror_std])

            n_active_list_alldrive.extend(n_active_list_drive)

            training_writer.add_scalar('val trans loss {}'.format(drive), terror_mean, cum_epoch)
            training_writer.add_scalar('val rot loss {}'.format(drive), rerror_mean, cum_epoch)

        if best_val_error > terror_mean:
            best_val_error = terror_mean
            log_tools.save_checkpoint_ccp(
                args.save_path, {
                    'epoch': epoch + 1 + pretrained_epoch,
                    'state_dict': mf_model.module.state_dict()
                }, True)


@torch.no_grad()
def validate(val_data_loader, flow_model, mf_model, args, drive, num_samples):
    # plt.rcParams.update({'font.size': 10})
    # fig = plt.figure(1)
    mf_h = round(args.height/args.downscale_f)
    mf_w = round(args.width/args.downscale_f)
    planar_depth = torch.ones(1, 1, mf_h, mf_w).to(device)
    n_active_neurons_list = []

    np.set_printoptions(precision=6, suppress=True)
    epe_drive = 0.0

    # run training for one epoch
    with torch.no_grad():
        poses_est = np.zeros((num_samples + 1, 6))
        poses_gt = np.zeros((num_samples + 1, 6))
        poses_mf = np.zeros((num_samples + 1, 6))
        times = np.zeros((num_samples + 1, 1))

        for i_batch, sample_batch in enumerate(val_data_loader):

            if i_batch < args.start:
                continue

            if i_batch > args.stop:
                print('Stopped after', args.stop, 'frames!')
                break

            # we do not need RGB for training
            times[i_batch + 1, :] = sample_batch['id']
            image_t = sample_batch['img_t'].to(device)
            image_tplus1 = sample_batch['img_tplus1'].to(device)

            # we need only flow for training, setting any nan entry to zero
            flow_gt = sample_batch['flow'].to(device)
            invalid = torch.isnan(flow_gt[:, 0, :, :]) + torch.isnan(flow_gt[:, 1, :, :])
            flow_gt[torch.stack((invalid, invalid), 1)] = 0.0

            flow_pwc = flow_model(image_t, image_tplus1)

            # we need the depth and pose GT only for validation
            depth_gt = sample_batch['depth'].to(device)
            pose_gt_r = sample_batch['pose_r'].to(device)
            pose_gt_t = sample_batch['pose_t'].to(device)

            # because pwc downscales by 4.0
            K = sample_batch['K'].float().to(device)
            fx = K[0, 0, 0] / args.downscale_f
            fy = K[0, 1, 1] / args.downscale_f
            A, B = motion_field.motion_mat_fxfy(mf_h, mf_w, fx, fy)

            mfg_output = mf_model(flow_pwc)
            pred_t_mf = mfg_output['t_mf']
            pred_r_mf = mfg_output['r_mf']
            latent_activations = torch.squeeze(mfg_output['out_conv5']).view(args.batch_size, -1)

            mft_gt = torch.zeros(args.batch_size, 2, mf_h, mf_w).cuda()
            mfw_gt = torch.zeros(args.batch_size, 2, mf_h, mf_w).cuda()

            t = pose_gt_t[0, :].contiguous().view(-1, 1)
            w = pose_gt_r[0, :].contiguous().view(-1, 1)
            vtx, vty, vwx, vwy = motion_field.gen_motion_field(t, 1 * w, torch.squeeze(planar_depth), A, B)
            mft_gt[0, 0, :, :] = vtx * fx
            mft_gt[0, 1, :, :] = vty * fy
            mfw_gt[0, 0, :, :] = vwx * fx
            mfw_gt[0, 1, :, :] = vwy * fy

            t_est = motion_field.est_translation(torch.squeeze(planar_depth[0, 0, :, :]), A, pred_t_mf[0, 0, :, :] / fx,
                                                 pred_t_mf[0, 1, :, :] / fy)
            r_est = motion_field.est_rotation(B, pred_r_mf[0, 0, :, :] / fx, pred_r_mf[0, 1, :, :] / fy)

            t_est_mf = motion_field.est_translation(torch.squeeze(planar_depth[0, 0, :, :]), A, mft_gt[0, 0, :, :] / fx,
                                                    mft_gt[0, 1, :, :] / fy)
            r_est_mf = motion_field.est_rotation(B, mfw_gt[0, 0, :, :] / fx, mfw_gt[0, 1, :, :] / fy)

            poses_est[i_batch + 1, :] = np.append(t_est.cpu().numpy(), r_est.cpu().numpy())
            poses_gt[i_batch + 1, :] = np.append(pose_gt_t.cpu().numpy(), pose_gt_r.cpu().numpy())
            poses_mf[i_batch + 1, :] = np.append(t_est_mf.cpu().numpy(), r_est_mf.cpu().numpy())

            if args.plot_data:
                epe_batch, n_active_neurons_batch = plot_minibatch(i_batch, sample_batch, flow_gt, flow_pwc, depth_gt, mft_gt, mfw_gt, pred_t_mf, pred_r_mf,
                               latent_activations)
                epe_drive += epe_batch
                n_active_neurons_list.append(n_active_neurons_batch)

        save_gt_pose_dir = args.save_path / drive / 'ground_truth'
        save_pred_pose_dir = args.save_path / drive / 'ours_prediction'
        save_mf_pose_dir = args.save_path / drive / 'mf'
        save_gt_pose_dir.makedirs_p()
        save_pred_pose_dir.makedirs_p()
        save_mf_pose_dir.makedirs_p()

        gt_file = save_gt_pose_dir / (drive+'.txt')
        pred_file = save_pred_pose_dir / (drive+'.txt')
        mf_file = save_mf_pose_dir / (drive+'.txt')

        rotation_lib.dump_pose_seq_cont(gt_file, poses_gt, times)
        rotation_lib.dump_pose_seq_cont(pred_file, poses_est, times)
        rotation_lib.dump_pose_seq_cont(mf_file, poses_mf, times)

        terror_mean, terror_std, rerror_mean, rerror_std = evaluate_rpe_tum.rpe_fun(gt_file, pred_file)
        # terror_mf_mean, terror_mf_std, rerror_mf_mean, rerror_mf_std = evaluate_rpe_tum.rpe_fun(gt_file, mf_file)
        return terror_mean, terror_std, rerror_mean, rerror_std, epe_drive, n_active_neurons_list


def train(train_data_loader, flow_model, mf_model, args, optimizer, training_writer):
    global n_iter
    train_loss = 0

    mf_h = round(args.height/args.downscale_f)
    mf_w = round(args.width/args.downscale_f)
    planar_depth = torch.ones(1, 1, mf_h, mf_w).to(device)

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(1)

    # run training for one epoch
    for i_batch, sample_batch in enumerate(train_data_loader):
        if i_batch * args.batch_size < args.start:
            continue

        if i_batch * args.batch_size > args.stop:
            print('Stopped after', args.stop, 'frames!')
            break

        # we do not need RGB for training
        image_t = sample_batch['img_t'].to(device)

        # we need only flow for training, setting any nan entry to zero
        flow_gt = sample_batch['flow'].to(device)
        invalid = torch.isnan(flow_gt[:, 0, :, :]) + torch.isnan(flow_gt[:, 1, :, :])
        flow_gt[torch.stack((invalid, invalid), 1)] = 0.0

        image_tplus1 = sample_batch['img_tplus1'].to(device)
        flow_pwc = flow_model(image_t, image_tplus1)

        # we need the depth and pose GT only for validation
        depth_gt = sample_batch['depth'].to(device)
        pose_gt_r = sample_batch['pose_r'].to(device)
        pose_gt_t = sample_batch['pose_t'].to(device)

        # because pwc downscales by 4.0
        K = sample_batch['K'].float().to(device)
        fx = K[0, 0, 0]/args.downscale_f
        fy = K[0, 1, 1]/args.downscale_f

        inflow = flow_pwc.detach().clone()

        mfg_output = mf_model(inflow)
        pred_t_mf = mfg_output['t_mf']
        pred_r_mf = mfg_output['r_mf']
        latent_activations = torch.squeeze(mfg_output['out_conv5']).view(args.batch_size, -1)

        mft_gt = torch.zeros(args.batch_size, 2, mf_h, mf_w).cuda()
        mfw_gt = torch.zeros(args.batch_size, 2, mf_h, mf_w).cuda()

        mft_loss = torch.zeros(args.batch_size, 1)
        mfw_loss = torch.zeros(args.batch_size, 1)
        sparsity_loss = torch.zeros(args.batch_size, 1)
        scale_t_loss = torch.zeros(args.batch_size, 1)
        scale_w_loss = torch.zeros(args.batch_size, 1)

        A, B = motion_field.motion_mat_fxfy(mf_h, mf_w, fx, fy)

        for im in range(args.batch_size):
            t = pose_gt_t[im, :].contiguous().view(-1, 1)
            # Convert right hand to left hand coordinates
            w = pose_gt_r[im, :].contiguous().view(-1, 1)
            vtx, vty, vwx, vwy = motion_field.gen_motion_field(t, 1 * w, torch.squeeze(planar_depth), A, B)
            mft_gt[im, 0, :, :] = vtx * fx
            mft_gt[im, 1, :, :] = vty * fy
            mfw_gt[im, 0, :, :] = vwx * fx
            mfw_gt[im, 1, :, :] = vwy * fy

            # MF reconstruction scale factor
            scale_t_loss[im, 0] = (torch.norm(mfw_gt[im, :, :, :], 2) / torch.norm(mft_gt[im, :, :, :], 2)).clamp(min=1, max=100)
            scale_w_loss[im, 0] = (torch.norm(mft_gt[im, :, :, :], 2) / torch.norm(mfw_gt[im, :, :, :], 2)).clamp(min=1, max=100)

            # MF reconstruction loss
            mft_loss[im, 0] = loss_functions.mf_loss_fun(pred_t_mf[im, :, :, :], mft_gt[im, :, :, :])
            mfw_loss[im, 0] = loss_functions.mf_loss_fun(pred_r_mf[im, :, :, :], mfw_gt[im, :, :, :])

            # Sparsity constraint loss
            sparsity_loss[im, 0] = loss_functions.sparsity_loss_gen_sigmoid(latent_activations[im, :])

        total_loss = torch.sum(scale_t_loss * mft_loss + scale_w_loss * mfw_loss + args.w_sparse * sparsity_loss)

        if torch.isnan(total_loss):
            print('loss is nan', n_iter)
            sys.exit()
        else:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        batch_sparsity_loss = torch.sum(sparsity_loss).item()
        batch_mft_loss = torch.sum(mft_loss).item()
        batch_mfw_loss = torch.sum(mfw_loss).item()
        batch_total_loss = torch.sum(total_loss).item()

        if args.log_freq > 0 and n_iter % args.log_freq == 0:
            training_writer.add_scalar('batch_translational_mf_loss', batch_mft_loss, n_iter)
            training_writer.add_scalar('batch_rotational_mf_loss', batch_mfw_loss, n_iter)
            training_writer.add_scalar('batch_sparsity_loss', batch_sparsity_loss, n_iter)

        if args.log_im_freq > 0 and n_iter % args.log_im_freq == 0:
            with torch.no_grad():
                true_mf_t_array = flowlib.flow_to_image(
                    printlib.gputensor2array(mft_gt[0, :, :, :])).transpose(2, 0, 1)
                true_mf_w_array = flowlib.flow_to_image(
                    printlib.gputensor2array(mfw_gt[0, :, :, :])).transpose(2, 0, 1)

                pred_mft_tplus1 = pred_t_mf
                pred_mfw_tplus1 = pred_r_mf
                pred_mf_t_array = flowlib.flow_to_image(
                    printlib.gputensor2array(pred_mft_tplus1[0, :, :, :])).transpose(2, 0, 1)
                pred_mf_w_array = flowlib.flow_to_image(
                    printlib.gputensor2array(pred_mfw_tplus1[0, :, :, :])).transpose(2, 0, 1)

                training_writer.add_image('Translational-mf: True v. Predicted',
                                          torchvision.utils.make_grid(
                                              [torch.tensor(true_mf_t_array),
                                               torch.tensor(pred_mf_t_array)]), n_iter)
                training_writer.add_image('Rotational-mf: True v. Predicted',
                                          torchvision.utils.make_grid(
                                              [torch.tensor(true_mf_w_array), torch.tensor(pred_mf_w_array)]), n_iter)

                flow_array = flowlib.flow_to_image(
                    printlib.gputensor2array(flow_gt[0, :, :, :])).transpose(2, 0, 1)
                inflow_array = flowlib.flow_to_image(
                    printlib.gputensor2array(inflow[0, :, :, :])).transpose(2, 0, 1)

                training_writer.add_image('GT flow',
                                          torchvision.utils.make_grid(
                                              [torch.tensor(flow_array)]), n_iter)
                training_writer.add_image('PWC flow',
                                          torchvision.utils.make_grid(
                                              [torch.tensor(torch.tensor(inflow_array))]), n_iter)
                del pred_mft_tplus1, pred_mfw_tplus1
        del pred_t_mf, pred_r_mf, mft_gt, mfw_gt, depth_gt, mfg_output, flow_gt, flow_pwc
        train_loss += batch_total_loss
        n_iter += 1
    return train_loss


@torch.no_grad()
def plot_minibatch(i_batch, sample_batch, flow_gt, flow_pwc, depth_gt, mft_gt, mfw_gt, mft_pred, mfw_pred, latent_activations):
    import cv2
    images = sample_batch['img_t']
    mst_mixed_inv = (latent_activations).cpu().reshape(-1, 1, 25, 40)
    epe_batch = 0.0
    n_active_neurons = 0

    for im in range(images.size(0)):
        depth_gt_im = depth_gt[im, 0, :, :].cpu().numpy()
        depth_gt_im = cv2.resize(depth_gt_im, (mft_gt.size(3), mft_gt.size(2)), interpolation=cv2.INTER_AREA)
        depth_gt_im = torch.tensor(depth_gt_im).cuda()

        mf_gt = (mft_gt[im, :, :, :] / depth_gt_im) + mfw_gt[im, :, :, :]
        mf_gt[:, depth_gt_im < 1e-5] = 0.0

        mf_pred = (mft_pred[im, :, :, :] / depth_gt_im) + mfw_pred[im, :, :, :]
        mf_pred[:, depth_gt_im < 1e-5] = 0.0

        flow_gt_im = refresh_transforms.rescale_flow(flow_gt[im], mf_gt.size(1), mf_gt.size(2)).to(device)

        invalid = torch.isnan(flow_gt_im[0, :, :]) + torch.isnan(flow_gt_im[1, :, :])

        obj_2d_gt = flow_gt_im - mf_gt
        obj_2d_pred = flow_gt_im - mf_pred

        th = 0.5

        object_mask_gt = object_motion_lib.find_object_mask_sintel(obj_2d_gt, invalid, th)
        object_scene_2d_gt = obj_2d_gt * torch.stack((object_mask_gt.float(), object_mask_gt.float()))

        object_mask_pred = object_motion_lib.find_object_mask_sintel(obj_2d_pred, invalid, th)
        object_scene_2d_pred = obj_2d_pred * torch.stack((object_mask_pred.float(), object_mask_pred.float()))

        plt.clf()
        plt.suptitle(str(i_batch * images.size(0) + im), fontsize=16)

        pl = plt.subplot(6, 4, 1)
        pl.imshow(printlib.cputensor2array(images[im, :, :, :]))
        pl.set_title("t")
        pl.set_xticks([])
        pl.set_yticks([])

        fl = plt.subplot(6, 4, 5)
        img = flowlib.flow_to_image(printlib.gputensor2array(flow_gt[im]))
        fl.imshow(img)
        fl.set_title("GT Flow")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 6)
        img = flowlib.flow_to_image(printlib.gputensor2array(flow_pwc[im]*4))
        fl.imshow(img)
        fl.set_title("PWC Flow")
        fl.set_xticks([])
        fl.set_yticks([])

        dm = plt.subplot(6, 4, 9)
        # dm.imshow(printlib.gpudepth2invarray(depth_t[im, :, :]))
        dm.imshow(torch.squeeze(depth_gt[im, :, :]).cpu().numpy(), cmap='hot')
        # hist, bin_edges = np.histogram(torch.squeeze(depth_t[im, :, :]).cpu().numpy(), bins=100)
        # dm.bar(bin_edges[:-1], hist, width=1)
        # #mask = depthmap < 0.5
        # #dm.imshow(mask)
        dm.set_title("Depth map t")
        dm.set_xticks([])
        dm.set_yticks([])

        fl = plt.subplot(6, 4, 3)
        img = flowlib.flow_to_image(printlib.gputensor2array(mf_gt))
        fl.imshow(img)
        fl.set_title("GT MF")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 4)
        img = flowlib.flow_to_image(printlib.gputensor2array(mf_pred))
        fl.imshow(img)
        fl.set_title("Predicted MF")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 7)
        img = flowlib.flow_to_image(printlib.gputensor2array(mft_gt[im, :, :, :]))
        fl.imshow(img)
        fl.set_title("GT MF-t")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 8)
        img = flowlib.flow_to_image(printlib.gputensor2array(mft_pred[im, :, :, :]))
        fl.imshow(img)
        fl.set_title("Predicted MF-t")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 11)
        img = flowlib.flow_to_image(printlib.gputensor2array(mfw_gt[im, :, :, :]))
        fl.imshow(img)
        fl.set_title("GT MF-w")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 12)
        img = flowlib.flow_to_image(printlib.gputensor2array(mfw_pred[im, :, :, :]))
        fl.imshow(img)
        fl.set_title("Predicted MF-w")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 15)
        img = flowlib.flow_to_image(printlib.gputensor2array(obj_2d_gt))
        fl.imshow(img)
        fl.set_title("GT Object Res")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 16)
        img = flowlib.flow_to_image(printlib.gputensor2array(obj_2d_pred))
        fl.imshow(img)
        fl.set_title("Predicted Object Res")
        fl.set_xticks([])
        fl.set_yticks([])

        act = plt.subplot(6, 4, 18)
        img = np.squeeze(mst_mixed_inv[im, :, :, :])
        act.imshow(img, cmap='gray')
        act.set_title("Latent activations")
        act.set_xticks([])
        act.set_yticks([])

        n_active_neurons += round(torch.sum(img > 0.01).item())

        pl = plt.subplot(6, 4, 19)
        mask = object_mask_gt.cpu().numpy()
        masked = np.ma.masked_where(mask == 0, mask)
        img = printlib.cputensor2array(refresh_transforms.rescale_img(images[im, :, :, :], mask.shape[0], mask.shape[1]))
        pl.imshow(img, interpolation='none')
        pl.imshow(masked, 'hsv', interpolation='none', alpha=0.5)
        #pl.imshow(object_mask_gt, cmap='gray')
        pl.set_xticks([])
        pl.set_yticks([])
        pl.set_title("GT object mask")

        pl = plt.subplot(6, 4, 20)
        mask = object_mask_pred.cpu().numpy()
        masked = np.ma.masked_where(mask == 0, mask)
        img = printlib.cputensor2array(
            refresh_transforms.rescale_img(images[im, :, :, :], mask.shape[0], mask.shape[1]))
        pl.imshow(img, interpolation='none')
        pl.imshow(masked, 'hsv', interpolation='none', alpha=0.5)
        # pl.imshow(object_mask_gt, cmap='gray')
        pl.set_xticks([])
        pl.set_yticks([])
        pl.set_title("Pred object mask")

        fl = plt.subplot(6, 4, 23)
        img = flowlib.flow_to_image(printlib.gputensor2array(object_scene_2d_gt))
        fl.imshow(img)
        fl.set_title("GT scene object")
        fl.set_xticks([])
        fl.set_yticks([])

        fl = plt.subplot(6, 4, 24)
        img = flowlib.flow_to_image(printlib.gputensor2array(object_scene_2d_pred))
        fl.imshow(img)
        fl.set_title("Pred scene object")
        fl.set_xticks([])
        fl.set_yticks([])

        tu = object_scene_2d_gt[0].cpu().numpy()
        tv = object_scene_2d_gt[1].cpu().numpy()
        u = object_scene_2d_pred[0].cpu().numpy()
        v = object_scene_2d_pred[1].cpu().numpy()

        epe = flowlib.flow_error(tu, tv, u, v)
        # print('epe: ', epe, ' #active neurons: ', n_active_neurons)
        epe_batch += epe

        plt.draw()
        plt.pause(5)
    return epe_batch, n_active_neurons


if __name__ == "__main__":
    main()
