"""
if you find these codes helpful, please cite our CVPR paper:
@inproceedings{zeng2022ideanet,
  title={IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment},
  author={Zeng, Yiming and Qian, Yue and Zhang, Qijian and Hou, Junhui and Yuan, Yixuan and He, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
also check and cite our previous CVPR paper: 
@inproceedings{zeng2020corrnet3d,
    author={Zeng, Yiming and Qian, Yue and Zhu, Zhiyu and Hou, Junhui and Yuan, Hui and He, Ying},
    title={CorrNet3D: Unsupervised End-to-end Learning of Dense Correspondence for 3D Point Clouds},
    booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021}
}

This is the inference demo for swing and longdress on DHB dataset.

install the env according to the requirements.txt: 
conda create --name <env> --file conda_requirements.txt

and compile the p1_EMDloss by "python setup.py install" 
or read the readme in the folder of p1_EMDloss:

cd ideanet_folder
conda activate ideanet
python inf_whole.py

#this will create "cache" folder under the ideanet_folder, 
#which contains interpolation inference results on 
#swing and longdress and their corresponding gt (in the format of .xyz point cloud files)
#and the mean EMD and CD metric as well as the per-frame loss xlsx
"""

#common packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader

#installed packages
import pandas as pd
import cv2
import ctypes as ct
from easydict import EasyDict as edict
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import yaml
import xlsxwriter

#inf fixed seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PRINT_ONCE = True

class RmOutlier:
    @staticmethod
    def from_o3dpcd_to_np(pcd):
        # convert Open3D.o3d.geometry.PointCloud to numpy array
        xyz_load = np.asarray(pcd.points)
        return xyz_load

    @staticmethod
    def from_np_to_o3dpcd(npxyz): 
        # Pass xyz to Open3D.o3d.geometry.PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npxyz)
        return pcd

    @staticmethod
    def _o3drmoutlier(npxyz):
        pcd = RmOutlier.from_np_to_o3dpcd(npxyz)
        cl,ind = pcd.remove_radius_outlier(nb_points=8, radius=0.091)
        xyz_ = RmOutlier.from_o3dpcd_to_np(cl)
        return xyz_

class chamfer_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred, gt):
        '''
        Input:
            pred: [B,N,3]
            gt: [B,N,3]
        '''
        chamfer_dist, _ = chamfer_distance(pred, gt) #input is BN3 BN3
        return chamfer_dist

class EMDloss(nn.Module):
    def __init__(self):
        super().__init__()
        import sys
        sys.path.append('./p1_EMDloss')
        from emd import earth_mover_distance_raw
        global earth_mover_distance_raw

    @staticmethod
    def cal_emd(P1, P2): #[B, 1024, 3] #[B, 1024, 3]
        N1, N2 = P1.size(1), P2.size(1)
        dist = earth_mover_distance_raw(P1, P2, transpose=False)
        emd_loss = (dist / N1).mean()
        return emd_loss   

    def forward(self , pc_gen, pc_gt):
        loss = EMDloss.cal_emd(pc_gen, pc_gt)
        return loss

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.args=args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type in ['dEMD']:
                loss_function = EMDloss()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )


        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.5f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        device = torch.device('cpu' if args.no_cuda else 'cuda')
        self.loss_module.to(device)

    def forward(self, RD, in_dict=None): #Return Dict
        loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None: 
                if l['type'] == 'dEMD': 
                    if hasattr(self.args, 'mssg') and self.args.mssg['dual_fuse']==True:
                        _loss=[]
                        for k_ in range(self.args.interframes):
                            _loss.append(l['function'](RD['only_i2o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                        _loss = sum(_loss) / len(_loss)  
                    elif hasattr(self.args, 'mssg') and self.args.mssg['only_one_branch']==True:
                        _loss=[]
                        for k_ in range(self.args.interframes):
                            _loss.append(l['function'](RD['i2o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                        _loss = sum(_loss) / len(_loss)             
                    else:
                        _loss1=[]
                        for k_ in range(self.args.interframes):
                            _loss1.append(l['function'](RD['i3o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                        _loss1 = sum(_loss1) / len(_loss1)
                        _loss2=[]
                        for k_ in range(self.args.interframes):
                            _loss2.append(l['function'](RD['i2o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                        _loss2 = sum(_loss2) / len(_loss2)
                        _loss = 0.5*(_loss1 + _loss2)

                effective_loss = l['weight'] * _loss
                losses[l['type']] = effective_loss
                loss += effective_loss
                
        return loss, losses

class Metric(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_of_o = self.args.interframes
        self.metric = []
        print('Preparing metric function:')
        for metric in args.metric.split('+'):
            weight, metric_type = metric.split('*')
            if metric_type in ['EMD']:
                metric_function = EMDloss()
            elif metric_type in ['CD']:
                metric_function = chamfer_loss()

            self.metric.append({
                'type': metric_type,
                'weight': float(weight),
                'function': metric_function}
            )

        if len(self.metric) > 1:
            self.metric.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.metric:
            if l['function'] is not None:
                print('{:.5f} * {}'.format(l['weight'], l['type']))

    def __call__(self, ID, RD):
        metric = 0
        metrics = {}
        for i, l in enumerate(self.metric):
            if l['function'] is not None: 
                if l['type'] == 'EMD': 
                    _metric=[]
                    for k_ in range(self.args.interframes):
                        _metric.append(l['function'](RD['i3o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                    _metric = sum(_metric) / len(_metric)
                    _metric = _metric.item()  
                    print(_metric)
                if l['type'] == 'CD':
                    _metric=[]
                    for k_ in range(self.args.interframes):
                        _metric.append(l['function'](RD['i3o{}'.format(k_+1)], RD['gt{}'.format(k_+1)]))
                        
                    _metric = sum(_metric) / len(_metric)
                       
                    _metric = _metric.item()   
                effective_metric = l['weight'] * _metric
                metrics[l['type']] = effective_metric
                metric += effective_metric
        return metric, metrics

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class DHBtest(Dataset):
    def __init__(self, args,cate): 
        self.args=args
        self.data_root = './'
        self.interframes = self.args.interframes #3
        self.cate = cate 
        self.data={}
        self.acc_len={}
        accu_len = 0
        self.cate_list_8ivfb = ['longdress','loot', 'redandblack', 'soldier']    
        for each_cat in [self.cate]:
            last_accu_len = accu_len
            if each_cat in self.cate_list_8ivfb:
                self.data[each_cat] = self.get_rich_data_8ivfb(each_cat)
            else:
                self.data[each_cat] = self.get_rich_data(each_cat)
            sample_len = self.data[each_cat][-1]
            accu_len += sample_len
            self.acc_len[each_cat]=[accu_len,last_accu_len]
        self.how_many_groups_in_total = accu_len

    def get_rich_data(self, cate):
        data_tensor = torch.load( os.path.join(self.data_root,cate+'_fps1024_aligned.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_group_index(len(data_tensor))
        print('cate ====',cate, 'len seq ====', len(data_tensor))
        return [data_tensor,GroupIdx,sample_len]

    def get_rich_data_8ivfb(self, cate):
        _path = './'
        data_tensor = torch.load( os.path.join(_path,cate+'.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_group_index(len(data_tensor))
        print('cate ====',cate, 'len seq ====', len(data_tensor))
        return [data_tensor,GroupIdx,sample_len]

    def get_one_group_index(self, len_, skips_=0):
        GroupIdx={}
        GroupIdx['skip']=[]
        GroupIdx['src']=[]
        GroupIdx['tgt']=[]
        for k in range(self.interframes):
            GroupIdx[f'gt{k}']=[]
        for i in range(len_):
            if i==0:
                src_idx=i 
            elif i>0:
                src_idx=tgt_idx;  
            how_many_skips = (self.interframes+1)
            tgt_idx=src_idx+self.interframes+1+how_many_skips*skips_
            if tgt_idx>=len_:
                break 
            else: 
                pass 
            GroupIdx['skip'].append(skips_);
            GroupIdx['src'].append(src_idx);
            GroupIdx['tgt'].append(tgt_idx);
            for s in range(self.interframes):
                mid_idx = src_idx + (s+1)*(skips_+1) 
                GroupIdx[f'gt{s}'].append(mid_idx)
        assert(len(GroupIdx['src'])==len(GroupIdx['tgt']))
        assert(len(GroupIdx['src'])==len(GroupIdx['gt0']))
        assert(len(GroupIdx['src'])==len(GroupIdx['skip']))
        sample_len = len(GroupIdx['src'])
        return GroupIdx,sample_len

    def random_permute(self, xyz, index):
        xyz = xyz.numpy()
        xyz_dict = {}
        npoint=xyz.shape[0]
        I=np.eye(npoint)
        p=I.copy()
        np.random.seed()
        while(np.array_equal(p,I)):
            np.random.seed(index) 
            np.random.shuffle(p) 
        permute_matrix = p
        permuted_xyz = np.dot(permute_matrix,xyz)
        xyz_dict['ori_verts'] = xyz
        xyz_dict['per_verts'] = permuted_xyz
        xyz_dict['perm'] = permute_matrix
        return permuted_xyz, permute_matrix

    def pc_normalize(self, pc, max_for_the_seq):
        pc = pc.numpy()
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max_for_the_seq
        pc = pc / m
        return pc

    def __getitem__(self, index):
        sample={}
        sample['indices']=[]
        sample['skip']=[]
        sample['ori_off_fpath']={}
        for k,v in self.acc_len.items():
            if index<v[0]:
                [data_tensor,GroupIdx,sample_len] = self.data[k]
                sample['cate']=k
                inside_idx = index-v[1]
                for kk,vv in GroupIdx.items():        
                    if kk=='skip':
                        ind_skip_val = GroupIdx['skip'][inside_idx]
                        sample['skip'].append(ind_skip_val)
                    elif kk!='skip': 
                        ind_val = vv[inside_idx]
                        sample['indices'].append(ind_val) #filled indices
                        if self.cate not in self.cate_list_8ivfb:                    
                            pth_str = ''
                            sample['ori_off_fpath'][kk] = pth_str
                        my_pc = data_tensor[ind_val]
                        if sample['cate'] in self.cate_list_8ivfb:
                            my_pc = self.pc_normalize(my_pc, max_for_the_seq=583.1497484423953)
                            my_pc = torch.from_numpy(my_pc)
                        sample[kk],sample[kk+'_perm']=self.random_permute(my_pc,ind_val) #filled pc
                sample['indices'] = np.array(sample['indices'])
                sample['skip'] = np.array(sample['skip'])
                return sample
                
    def __len__(self):
        return self.how_many_groups_in_total

    @staticmethod
    def DHBtestLoader(args,shuffle=False,cate=''):
        dataset = DHBtest(args,cate)
        dloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        print('DHBtest cat: ====',cate)
        print('DHBtest len: ====',len(dataset))
        print('DHBtest Batch size: ====',args.batch_size)
        print('num of Batches: ====',len(dloader))
        return dloader

def get_norm_layer(out_channels, layer_type='GroupNorm'):
    '''https://discuss.pytorch.org/t/how-to-write-different-layers-setting-in-sequential/34370/2'''
    if layer_type == 'BatchNorm2d':
        return nn.BatchNorm2d(out_channels)
    elif layer_type == 'BatchNorm1d':
        return nn.BatchNorm1d(out_channels)
    elif layer_type == 'GroupNorm':
        return nn.GroupNorm(out_channels//16, out_channels)

class DistP(object):
    def __init__(self):
        pass

    def square_distances(self,P, Q):
        # P: [B, N, C]
        # Q: [B, M, C]
        assert P.size(0)==Q.size(0) and P.size(2)==Q.size(2)
        B, N, _ = P.shape
        _, M, _ = Q.shape
        D = -2 * torch.matmul(P, Q.permute(0, 2, 1))
        D += torch.sum(P**2, -1).view(B, N, 1)
        D += torch.sum(Q**2, -1).view(B, 1, M)
        return torch.abs(D) # [B, N, M]

    def cal_p_for_f3d(self, xyz1, xyz2):
        #B N1 3, B N2 3
        dist_matrix = self.square_distance(xyz1, xyz2) #B N1 N2
        similarity = 1.0 / (dist_matrix+1e-13)
        p = similarity
        return p

class Encoder(nn.Module): 
    def __init__(self, args, enc_emb_dim=128, enc_glb_dim=512, k_nn=20):
        super(Encoder, self).__init__()
        self.args =args
        self.k = k_nn
        norm_type = 'BatchNorm2d'
        self.bn1 = get_norm_layer(out_channels=64, layer_type=norm_type)
        self.bn2 = get_norm_layer(out_channels=64, layer_type=norm_type)
        self.bn3 = get_norm_layer(out_channels=128, layer_type=norm_type)
        self.bn4 = get_norm_layer(out_channels=256, layer_type=norm_type)
        
        self.bn5 = nn.GroupNorm(32,enc_glb_dim//2)
        self.bn6 = nn.GroupNorm(32,512)
        self.bn7 = nn.GroupNorm(32,256)
        self.bn8 = nn.GroupNorm(32,enc_emb_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),self.bn4,nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, enc_glb_dim//2, kernel_size=1, bias=False),self.bn5,nn.LeakyReLU(negative_slope=0.2))
     
        self.mlp = nn.Sequential(   
            nn.Conv1d(64+64+128+256+enc_glb_dim, 512, 1),
            self.bn6,
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            self.bn7,
            nn.ReLU(),
            nn.Conv1d(256, enc_emb_dim, 1),
            self.bn8,
            nn.ReLU())

    def _get_graph_feature(self, x, k=20, idx=None):

        def knn(x, k): 
            inner = -2*torch.matmul(x.transpose(2, 1), x) 
            xx = torch.sum(x**2, dim=1, keepdim=True)  
            pairwise_distance = -xx - inner - xx.transpose(2, 1)  
            idx = pairwise_distance.topk(k=k, dim=-1)[1]  
            return idx

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points) 
        
        if idx is None:
            idx = knn(x, k=k)   

        device = idx.device 

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()  
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, transpose_xyz):  #B 3 N
        x = transpose_xyz
        batch_size = x.size(0)
        num_points=x.size(2)
        
        x = self._get_graph_feature(x, self.k)  
   
        x = self.conv1(x) 
        x1 = x.max(dim=-1, keepdim=False)[0]  #torch.Size([1, 64, 1024])
        x = self._get_graph_feature(x1, self.k) 
        x = self.conv2(x) 
        x2 = x.max(dim=-1, keepdim=False)[0]  #torch.Size([1, 64, 1024])
        x = self._get_graph_feature(x2, self.k)
        x = self.conv3(x) 
        x3 = x.max(dim=-1, keepdim=False)[0]  #torch.Size([1, 128, 1024])
        x = self._get_graph_feature(x3, self.k) 
        x = self.conv4(x) 
        x4 = x.max(dim=-1, keepdim=False)[0]  #torch.Size([1, 256, 1024])
        x = torch.cat((x1, x2, x3, x4), dim=1) 
        local_concat = x 
        x = self.conv5(x) 
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)                      
        global_vector = x                               
        repeat_glb_feat = global_vector.unsqueeze(-1).expand(batch_size, global_vector.shape[1], num_points)
        x = torch.cat((local_concat, repeat_glb_feat), 1)  
        embedding_feat = self.mlp(x)                   
        return embedding_feat, global_vector.unsqueeze(-1)  

class norm(nn.Module):
    def __init__(self, axis=2):
        super(norm, self).__init__()
        self.axis = axis

    def forward(self, x): #torch.Size([B=10, 1024, 1024]) N1 N2
        mean = torch.mean(x, self.axis,keepdim=True) #torch.Size([B=10, 1024])
        std = torch.std(x, self.axis,keepdim=True)   #torch.Size([B=10, 1024])
        x = (x-mean)/(std+1e-6) #torch.Size([10, 1024, 1024])   
        return x

class Gradient(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x

class CorrNet3D(nn.Module):

    def __init__(
        self,
        args,
        input_pts: int = 1024,
        k_nn: int = 20,
        enc_emb_dim: int = 128,
        enc_glb_dim: int = 512,
        # ls_coeff: list = [10.0, 1.0, 0.1],
        dec_in_dim: int = 1024+3
    ):
        super(CorrNet3D, self).__init__()
        self.args = args
        if self.args.num_points != None:
            input_pts=self.args.num_points
        
        self.input_pts = input_pts
        print(self.input_pts)
        self.enc_emb_dim = enc_emb_dim
        if self.args.enc_glb_dim !=None:
            enc_glb_dim = self.args.enc_glb_dim
        self.enc_glb_dim = enc_glb_dim
        self.dec_in_dim = enc_glb_dim + 3
        self.k_nn = k_nn
        
        self.encoder  = Encoder(args, self.enc_emb_dim, self.enc_glb_dim, self.k_nn)

        self.DeSmooth = nn.Sequential(
            nn.Conv1d(in_channels=self.input_pts, out_channels=self.input_pts+128, kernel_size=1, stride=1,  bias=False),
            nn.ReLU(), 
            norm(axis=1),
            nn.Conv1d(in_channels=self.input_pts+128, out_channels=self.input_pts, kernel_size=1, stride=1,bias=False),
            Modified_softmax(axis=2) 
            ) 
        norm_type='BatchNorm1d'
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=self.dec_in_dim, out_channels=self.dec_in_dim,      kernel_size=1),
            get_norm_layer(out_channels=self.dec_in_dim, layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim, out_channels=self.dec_in_dim//2, kernel_size=1),
            get_norm_layer(out_channels=self.dec_in_dim//2, layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim//2, out_channels=self.dec_in_dim//4, kernel_size=1),
            get_norm_layer(out_channels=self.dec_in_dim//4, layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.dec_in_dim//4, out_channels=3, kernel_size=1),
            nn.Tanh(),
            ) 

    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) 
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) 
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P

        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity=-pairwise_distance
        idx = similarity.topk(k=k, dim=-1)[1]
        return pairwise_distance, idx
    
    def forward(self, xyz1, xyz2): 
        HIER_feat1, pooling_feat1 = self.encoder(transpose_xyz=xyz1.transpose(1, 2)) 
        HIER_feat2, pooling_feat2 = self.encoder(transpose_xyz=xyz2.transpose(1, 2))   
        self.forward_pooling_feat1 = pooling_feat1
        self.forward_pooling_feat2 = pooling_feat2   
        pairwise_distance, _ = self._KFNN(HIER_feat1, HIER_feat2)
        similarity = 1/(pairwise_distance + 1e-6) #NsxNt
        p = self.DeSmooth(similarity.transpose(1,2).contiguous()).transpose(1,2).contiguous() 

        return p 
  
class IntpFeat(nn.Module):
    def __init__(self,args):
        super().__init__() 
        self.args = args
        self.cal_p = DistP()
        self.no_cuda = self.args.no_cuda
        self.print_one = True
        self.init_learnable_layers() 

    def normalize_point_cloud(self,input):
        """
        input: pc [N, P, 3]
        output: pc, centroid, furthest_distance
        """
        if len(input.shape) == 2:
            axis = 0
        elif len(input.shape) == 3:
            axis = 1
        centroid = np.mean(input, axis=axis, keepdims=True)

        input = input - centroid
        furthest_distance = np.amax(
            np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)

        scale = furthest_distance
        input = input / scale
        return torch.from_numpy(input), torch.from_numpy(centroid), torch.from_numpy(scale)
       
    def init_learnable_layers(self):
        self.out_num = self.args.interframes 
        self.corrlrner = CorrNet3D(self.args,
            input_pts = self.args.num_points,
        k_nn = 20,
        enc_emb_dim = 128,
        enc_glb_dim = 512,
        dec_in_dim = self.args.num_points+3)
        self.feat = Encoder(self.args, enc_emb_dim=128, enc_glb_dim=self.args.enc_glb_dim, k_nn=20) #1024
        self.Decoder = self._decoder()
        self.criterion = Loss(self.args)
        self.inf_metric = Metric(self.args)

    def _decoder(self):

        d_=128+self.args.enc_glb_dim
        dim_=[d_, d_, d_//2, d_//4, self.out_num]
        norm_type='BatchNorm1d'
        layers = nn.Sequential(
            nn.Conv1d(in_channels=dim_[0], out_channels=dim_[1],      kernel_size=1),
            get_norm_layer(out_channels=dim_[1], layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=dim_[1], out_channels=dim_[2], kernel_size=1),
            get_norm_layer(out_channels=dim_[2], layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=dim_[2], out_channels=dim_[3], kernel_size=1),
            get_norm_layer(out_channels=dim_[3], layer_type=norm_type),
            nn.ReLU(),
            nn.Conv1d(in_channels=dim_[3], out_channels=dim_[4], kernel_size=1),
            nn.Tanh(),
            )
        return layers   

    def BatchMatrixMult(self, matrix1, matrix2): #BAC BCD = BAD
        return torch.bmm(matrix1, matrix2)

    def getDualAlignEmbedding(self, featSrc, featTgt, glbSrc, glbTgt, P_st, P_ts, t): #BDN BDN scalar
        """BDNs BDNt BF1 BF1 BNsNt BNtNs scalar"""
        #Src Order Branch
        AlignedTgtasSrc = self.BatchMatrixMult(P_st, featTgt.transpose(1, 2)).transpose(1, 2) #BDNs same order as src
        intplFeatAlignedasSrc = self.getTimeEmbedding(feat_left=featSrc, feat_right=AlignedTgtasSrc, t=t) #BDNs BDNs #local-feat #ft;weight_acc
        intplglb = self.getTimeEmbedding(feat_left=glbSrc, feat_right=glbTgt, t=t) #glb-feat
        mix_feat_alignedasSrc = self._local_glb_feature_mixer(intplFeatAlignedasSrc, intplglb) 
        #Tgt Order Branch  
        AlignedSrcasTgt = self.BatchMatrixMult(P_ts, featSrc.transpose(1, 2)).transpose(1, 2) #BNtNs @ BNsD = BNtD - > BDNt same order as tgt
        intplFeatAlignedasTgt = self.getTimeEmbedding(feat_left=AlignedSrcasTgt, feat_right=featTgt, t=t) #BDNt BDNt#local-feat
        intplglb = self.getTimeEmbedding(feat_left=glbSrc, feat_right=glbTgt, t=t)#glb-feat
        mix_feat_alignedasTgt = self._local_glb_feature_mixer(intplFeatAlignedasTgt, intplglb)
        return mix_feat_alignedasSrc, mix_feat_alignedasTgt

    def getDecoderVal(self, embSrc, embTgt):
        return self.Decoder(embSrc), self.Decoder(embTgt) #[Bx3xN]

    def _local_glb_feature_mixer(self, local_feat, global_feat):
        B_, D_, N_ = local_feat.shape #14x128x1024
        global_feat = global_feat.repeat(1,1, N_) #BFN
        mix_feat = torch.cat((local_feat,global_feat),dim=1) #B(D+F)N
        return mix_feat

    def getTimeEmbedding(self, feat_left, feat_right, t):
        f_t = (1-t)*feat_left + t*feat_right 
        return f_t

    def getOneIntp(self,src,tgt,P_st, P_ts, t):
        Intermediate_results={}
        featSrc, glbSrc = self.feat(src.transpose(1, 2)) #f_i2=[Bx128x1024] pooling_feat2=BxFx1
        featTgt, glbTgt = self.feat(tgt.transpose(1, 2)) #f_i3=[Bx128x1024]BCN pooling_feat3=BxFx1  

        mix_feat_alignedasSrc, mix_feat_alignedasTgt = self.getDualAlignEmbedding(featSrc, featTgt, glbSrc, glbTgt, P_st, P_ts, t) #Bx(1024+128)xNs Bx(1024+128)xNt
        dSrc, dTgt = self.getDecoderVal(mix_feat_alignedasSrc, mix_feat_alignedasTgt) #[Bx3xN]
        middle_src=self.getTimeEmbedding(feat_left=src, \
                                            feat_right=self.BatchMatrixMult(P_st, tgt), \
                                            t=t) #local-points
        middle_tgt=self.getTimeEmbedding(feat_left=self.BatchMatrixMult(P_ts, src), \
                                            feat_right=tgt, \
                                            t=t) #local-points
        Os = middle_src + dSrc.transpose(1, 2) #BNs3
        Ot = middle_tgt + dTgt.transpose(1, 2) #BNt3  
        Intermediate_results['middle_src']=middle_src
        Intermediate_results['middle_tgt']=middle_tgt  
        Intermediate_results['delta_src']=dSrc.transpose(1, 2)  
        Intermediate_results['delta_tgt']=dTgt.transpose(1, 2)  
        output_pc = [Os, Ot]
        return output_pc,Intermediate_results

    def forward(self, ID):  
        T_list = [t for t in np.linspace(0.0, 1.0, num=self.args.interframes+2)]
        T_list = T_list[1:-1]
            
        TempOut = {}
        RD = {}
        global PRINT_ONCE
        i_2, i_3, perm2, perm3 = ID['i_2'], ID['i_3'], ID['perm2'], ID['perm3']

        if self.args.groupNorm==True: 
            i_2, centroid, scale = self.normalize_point_cloud(i_2.cpu().numpy()) #BN3, B13, B11
            i_2 = i_2.cuda()
            centroid = centroid.cuda()
            scale=scale.cuda()
            i_3 = (i_3-centroid)/scale

        P_ = self.corrlrner(i_2, i_3) #BxNi2xNi3 <- BxNi2x3 BxNi3x3 
        P_t = P_.transpose(1, 2)   
        for k in range(len(T_list)):
            TempOut['o{}'.format(k+1)], TempOut['IR{}'.format(k+1)] = self.getOneIntp(i_2, i_3, P_, P_t, T_list[k])

        if self.args.groupNorm==True: #de groupNorm for i_2 o1 o2 o3 i_3
            i_2 = i_2*scale + centroid
            i_3 = i_3*scale + centroid
            for k in range(len(T_list)):
                if type(TempOut['o{}'.format(k+1)]).__name__=='list':
                    assert(len(TempOut['o{}'.format(k+1)])==2) #o1
                    for kk in range(len(TempOut['o{}'.format(k+1)])):
                        TempOut['o{}'.format(k+1)][kk] = TempOut['o{}'.format(k+1)][kk]*scale + centroid
                else:
                    TempOut['o{}'.format(k+1)] = TempOut['o{}'.format(k+1)]*scale + centroid
        assert(isinstance(TempOut['o1'],list)==True)
        RD['src'] = i_2
        RD['tgt'] = i_3
        RD['P_'] = P_
        RD['P_t'] = P_t
        RD['gtP_'] = torch.bmm(perm2, perm3.permute(0,2,1))
        RD['gtP_t'] = torch.bmm(perm3, perm2.permute(0,2,1))
        for k in range(len(T_list)):
            RD['gt{}'.format(k+1)] = ID['gt{}'.format(k+1)]
            RD['i2o{}'.format(k+1)] = TempOut['o{}'.format(k+1)][0]
            RD['i3o{}'.format(k+1)] = TempOut['o{}'.format(k+1)][1]
            RD['i2d{}'.format(k+1)] = TempOut['IR{}'.format(k+1)]['delta_src']
            RD['i3d{}'.format(k+1)] = TempOut['IR{}'.format(k+1)]['delta_tgt']
            RD['i2m{}'.format(k+1)] = TempOut['IR{}'.format(k+1)]['middle_src']
            RD['i3m{}'.format(k+1)] = TempOut['IR{}'.format(k+1)]['middle_tgt']
        if self.args.timeVarPick:               
            if PRINT_ONCE:
                print('using timeVarPick')
                PRINT_ONCE = False
            if len(T_list)>=2:
                for k in range(len(T_list)):
                    if k < len(T_list)//2:
                        RD['i3o{}'.format(k+1)] = RD['i2o{}'.format(k+1)]
        return RD

    def get_metric(self, ID, RD): #return dict
        metric, metrics = self.inf_metric(ID, RD)
        return metric, metrics

class RunOneSeq(object):
    def __init__(self, args, infer_loader, name, cache_root, no_cuda):
        self.args = args 
        self.infer_loader = infer_loader
        self.name = name
        self.cache_root = cache_root
        self.no_cuda = no_cuda
        self.creating() 

    def creating(self):
        self.metric_buf = [] 
        self.metrics_buf = {}
        self.vis = {}
        self.vis['gen_P_dict'] = [] #
        self.vis['gen_seq'] = [] #
        self.vis['gt_seq'] = [] #
        self.ID={}
        self.RD={}
        self.emd_loss_func = EMDloss()
        self.cd_loss_func = chamfer_loss()

    def input_dict_filling_2(self, data_dict):
        if not self.no_cuda:
            self.ID={
                'i_2':data_dict['src'][:,:,:3].float().cuda(),
                'i_3':data_dict['tgt'][:,:,:3].float().cuda(),
                'perm2':data_dict['src_perm'].float().cuda(),
                'perm3':data_dict['tgt_perm'].float().cuda(),
            }
            for k in range(self.args.interframes):
                self.ID['gt{}'.format(k+1)] = data_dict['gt{}'.format(k)][:,:,:3].float().cuda()
                self.ID['permgt{}'.format(k+1)] = data_dict['gt{}_perm'.format(k)].float().cuda()
        else:
            self.ID={
                'i_2':data_dict['src'][:,:,:3].float(),
                'i_3':data_dict['tgt'][:,:,:3].float(),
                'perm2':data_dict['src_perm'].float(),
                'perm3':data_dict['tgt_perm'].float(),
            }
            for k in range(self.args.interframes):
                self.ID['gt{}'.format(k+1)] = data_dict['gt{}'.format(k)][:,:,:3].float()
                self.ID['permgt{}'.format(k+1)] = data_dict['gt{}_perm'.format(k)].float()

    def plot_seq_loss(self, path_name, out_seq, gt_seq):
        assert(len(out_seq)==len(gt_seq))
        workbook = xlsxwriter.Workbook(path_name)
        worksheet = workbook.add_worksheet()
        if self.args.VideoRmOutlier:
            worksheet.write('A1','VideoRmOutlier_frame')
            worksheet.write('B1','VideoRmOutlier_EMD')
            worksheet.write('C1','VideoRmOutlier_CD')
        else:
            worksheet.write('A1','frame')
            worksheet.write('B1','EMD')
            worksheet.write('C1','CD')
        for i in range(len(out_seq)):
            if self.args.VideoRmOutlier:
                out_item = RmOutlier._o3drmoutlier(out_seq[i]) 
                out_item = torch.from_numpy(out_item).unsqueeze(0).float()
            else:
                out_item = torch.from_numpy(out_seq[i]).unsqueeze(0)
            emd_loss = self.emd_loss_func(out_item.cuda(), torch.from_numpy(gt_seq[i]).unsqueeze(0).cuda())
            cd_loss = self.cd_loss_func(out_item.cuda(), torch.from_numpy(gt_seq[i]).unsqueeze(0).cuda())
            worksheet.write('A{}'.format(2+i), str(int(1+i)))
            worksheet.write('B{}'.format(2+i), str(emd_loss.item()))
            worksheet.write('C{}'.format(2+i), str(cd_loss.item()))
        workbook.close()

    def looping(self):
        print("len(self.infer_loader)",len(self.infer_loader))
        self.sampleCnt = 0
        # visSample_dirname = os.path.join(self.cache_root, f'{self.name}_visP')
        # os.makedirs(visSample_dirname, exist_ok=True)
        for iter, data_dict in tqdm(enumerate(self.infer_loader)):
            self.input_dict_filling_2(data_dict) 

            self.ID['in_infer_phase'] = True
            self.RD = self.model(self.ID) 
            bsize = self.RD['i3o1'].shape[0]
            for k in range(bsize):
                for k_ in range(self.args.interframes):
                    self.vis['gen_seq'].extend([self.RD['i3o{}'.format(k_+1)][k].cpu().numpy()])
                    self.vis['gt_seq'].extend([self.ID['gt{}'.format(k_+1)][k].cpu().numpy()])
            metric, metrics = self.model.get_metric(self.ID, self.RD)
            for key,value in metrics.items():
                if key not in self.metrics_buf:
                    self.metrics_buf[key]=[]
                self.metrics_buf[key].append(value)
            self.metric_buf.append(metric)

    def after_looping(self):
        print(f'{self.name} Avg metric {np.mean(self.metric_buf)}')
        for key,val in self.metrics_buf.items(): 
            self.metrics_buf[key] = np.mean(self.metrics_buf[key]) 
        print(f'{self.name} avg metric dict',self.metrics_buf) 
        ######## gather list
        single_sequence_metric_dict = {}
        single_sequence_metric_dict['Name']=self.name
        for key,val in self.metrics_buf.items():
            single_sequence_metric_dict[key] = val
        ########per frame loss
        if self.args.runOneSeqSaveOpt.saveXlsx==1:
            dirname = os.path.join(self.cache_root, f'{self.name}_Xlsx')
            os.makedirs(dirname, exist_ok=True)
            if self.args.VideoRmOutlier:
                name_ = os.path.join(dirname, f'VideoRmOutlier_EMDCD_{self.name}.xlsx')
            else:
                name_ = os.path.join(dirname, f'EMDCD_{self.name}.xlsx')
            self.plot_seq_loss(path_name=name_, out_seq=self.vis['gen_seq'], gt_seq=self.vis['gt_seq'])
        ########in betw output
        if self.args.runOneSeqSaveOpt.saveO==1:
            dirname = os.path.join(self.cache_root, self.name)
            os.makedirs(dirname, exist_ok=True)
            assert(len(self.vis['gen_seq'])==len(self.vis['gt_seq']))
            cnt = 0
            # outputs = []
            print(f'saving generated {self.name} seq')
            for item in tqdm(self.vis['gen_seq'][:len(self.vis['gen_seq'])//self.args.render_ratio]): #item.shape=1024x3
                if self.args.VideoRmOutlier:
                    item = RmOutlier._o3drmoutlier(item) 
                if self.args.runOneSeqSaveOpt.saveXYZ==True:
                    np.savetxt(os.path.join(dirname, 'img_{}.xyz'.format(cnt)), item)

                cnt+=1

        ########GT
        if self.args.runOneSeqSaveOpt.saveGT==1: #sometimes user can turn it false to save time
            dirname = os.path.join(self.cache_root, f'{self.name}_gt')
            os.makedirs(dirname, exist_ok=True)
            cnt = 0

            print(f'saving gt {self.name} seq')
            for item in tqdm(self.vis['gt_seq'][:len(self.vis['gt_seq'])//self.args.render_ratio]):
                if self.args.runOneSeqSaveOpt.saveXYZ==True:
                    np.savetxt(os.path.join(dirname, 'gt_img_{}.xyz'.format(cnt)), item)

                cnt+=1

        return single_sequence_metric_dict 

    def run_one_seq(self, model):
        self.model = model 
        self.looping() 
        single_sequence_metric_dict = self.after_looping()
        return single_sequence_metric_dict

class Inference(object):
    def __init__(self, args):
        self.args = args
        if args.exp_name != None:
            self.experiment_id = args.exp_name
        cache_root = 'cache/%s' % self.experiment_id

        os.makedirs(cache_root, exist_ok=True)
        sys.stdout = Logger(os.path.join(cache_root, 'log.txt'))
        self.cache_root = cache_root

        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.task = args.task

        print(vars(args))
        print('metric is {}'.format(args.metric))

        if self.task == 'interpolation_sequence':
            self.infer_loader = {}
            if 'swing' in args.select_seq:
                self.infer_loader['swing'] = DHBtest.DHBtestLoader(args,shuffle=False,cate='swing')
            if 'longdress' in args.select_seq:
                self.infer_loader['longdress'] = DHBtest.DHBtestLoader(args,shuffle=False,cate='longdress')
            self.AGENT={} 
            for k,v in self.infer_loader.items():
                self.AGENT[k] = RunOneSeq(args=args, infer_loader=v, name=k, cache_root=self.cache_root, no_cuda=self.no_cuda)
        # initialize model
        self.model  = IntpFeat(args)
        self._load_pretrain(args.model_path)
        # load model to gpu
        self.model = self.model.cuda()
        self.args = args
        fp = open(os.path.join(self.cache_root,'args.yaml'), 'w')
        fp.write(yaml.dump(vars(self.args)))
        fp.close()
        print('save args to: ', os.path.join(self.cache_root,'args.yaml'))

    def run_seq(self):
        all_list=[]
        name_list = ['Name', 'EMD', 'CD']
        for k,v in self.AGENT.items():
            self.model.eval()
            single_sequence_metric_dict = v.run_one_seq(self.model)
            single_list = []
            single_name_list = []
            for kk, vv in single_sequence_metric_dict.items():
                single_name_list.append(kk)
                if kk == 'EMD' or kk == 'CD':
                    my_vv = np.around(vv*np.float32(1000), 3)
                    single_list.append(my_vv)
                else:
                    single_list.append(vv)
            #name checking
            for i in range(len(single_name_list)):
                assert(single_name_list[i] == name_list[i])
            #pd list-list
            all_list.append(single_list)
        df = pd.DataFrame(all_list, columns = name_list)
        df.to_csv(os.path.join(self.cache_root, 'mean-metrics.csv'))
              
    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key=='best_metric':
                continue
            if key[:6] == 'module':
                name = key[7:] 
            else:
                name = key
            if key[:10] == 'classifier':
                continue
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")

with open(os.path.join("inf_whole_args.yaml"), 'r') as stream:
    parsed_yaml=yaml.load(stream, Loader=yaml.Loader)
    args = edict(parsed_yaml)
    
if __name__ == '__main__':   
    with torch.no_grad():
        inference = Inference(args)            
        inference.run_seq()      
