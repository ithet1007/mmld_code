
import torch
import torch.nn as nn
import numpy as np

# HNM_heatmap loss for heatmap regression
class HNM_heatmap(nn.Module):
    def __init__(self, R=20):
        super(HNM_heatmap, self).__init__()
        self.R = R
        self.regressionLoss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, heatmap, target_heatmap):
        loss = 0
        batch_size = heatmap.size(0)
        n_class = heatmap.size(1)
        heatmap = heatmap.reshape(batch_size, n_class, -1)
        target_heatmap = target_heatmap.reshape(batch_size, n_class, -1)
        for i in range(batch_size):
            for j in range(n_class):
                # counting the heatmap voxels
                select_number = torch.sum(
                    target_heatmap[i, j] >= 0).int().item()
                
                if select_number <= 0:
                    # if landmark is nonexist, setting a fixed number of hard negative mining
                    select_number = int(self.R * self.R * self.R / 8)
                else:
                    # if existing a landmark, regress these voxels inside the mask
                    _, cur_idx = torch.topk(
                        target_heatmap[i, j], select_number)
                    predict_pos = heatmap[i, j].index_select(0, cur_idx)
                    target_pos = target_heatmap[i, j].index_select(0, cur_idx)
                    loss += self.regressionLoss(predict_pos, target_pos)

                # using hard negative mining for background voxels
                # the default background voxel is -1
                mask_neg = 1 - target_heatmap[i, j]
                neg_number = torch.sum(
                    target_heatmap[i, j] < 0).int().item()
                _, neg_idx = torch.topk(mask_neg, neg_number)
                predict_neg = heatmap[i, j].index_select(0, neg_idx)
                _, cur_idx = torch.topk(predict_neg,
                                        select_number)
                predict_neg = heatmap[i, j].index_select(0, cur_idx)
                target_neg = target_heatmap[i, j].index_select(0, cur_idx)
                loss += self.regressionLoss(predict_neg, target_neg)
        return loss / (batch_size * n_class)


# HNM_propmap loss for yolol model training
class HNM_propmap(nn.Module):
    def __init__(self, n_class=14, lambda_hnm=0.2,lambda_noobj=0.001, device=None): #0.2
        super(HNM_propmap, self).__init__()
        self.regressionLoss = nn.SmoothL1Loss() # regression loss
        self.bceLoss = nn.BCEWithLogitsLoss() # classification loss
        self.n_class = n_class
        self.lambda_hnm = lambda_hnm # the weight for hard negative mining
        self.lambda_noobj = lambda_noobj # the weight for regularization to make background deactivate
        self.device = device
        self.hard_num = 256 # the selected number for nonexist landmark

    def forward(self, proposal_map, proposals):
        loss = 0
        batch_size = proposal_map.size(0)
        
        cl_pred_pos = []
        cl_pred_neg = []
        reg_pred = []
        reg_target = []
        hard_neg_count = np.zeros((self.n_class, )).astype("int32")
        hard_neg_pred = []
        for i in range(batch_size):
            for anchor_idx, proposal in enumerate(proposals[i]):
                for bbox in proposal:
                    c=int(bbox[0]); w=int(bbox[1]); h=int(bbox[2])
                    # -100 indicate the padded proposal
                    # the details refer to class LandmarkProposal in data_utils/transforms.py
                    if bbox[-1] == -100:
                        break
                    elif bbox[-1] >= 0:
                        # if landmark exist, generate prediction and target of relative coordinates
                        cl_pred_pos.append(proposal_map[i, c, w, h, anchor_idx, int(3+bbox[-1]):int(4+bbox[-1])])  
                        cl_pred_neg.append(proposal_map[i, c, w, h, anchor_idx, 3:int(3+bbox[-1])])  
                        cl_pred_neg.append(proposal_map[i, c, w, h, anchor_idx, int(4+bbox[-1]):])  
                        reg_pred.append(proposal_map[i, c, w, h, anchor_idx, :3])       
                        reg_target.append(bbox[3:-1])
                    else:
                        # if landmark nonexist, indicate the label for hard negative mining
                        hard_neg_count[-1-int(bbox[-1].item())] += 1

        # select hard negative voxels for nonexist landmarks
        for i in range(self.n_class):
            if hard_neg_count[i] != 0:
                cur_negative = proposal_map[:,:,:,:,:,3+i].reshape(-1)
                _, neg_idx = torch.topk(cur_negative, hard_neg_count[i]*self.hard_num)
                hard_neg_pred.append(cur_negative[neg_idx])


        cl_pred_pos = torch.cat(cl_pred_pos, 0)
        cl_pred_neg = torch.cat(cl_pred_neg, 0)
        ################## classification loss for positive ############################
        cl_pos_loss= self.bceLoss(cl_pred_pos, torch.ones((cl_pred_pos.shape[0],)).to(self.device))
        ################## classification loss for negative ######################
        cl_neg_loss= 1/(self.n_class-1) * self.bceLoss(cl_pred_neg, torch.zeros((cl_pred_neg.shape[0],)).to(self.device))

        ################# classification loss for hard negative #########################
        cl_hard_neg_loss = 0
        if len(hard_neg_pred) > 0:
            hard_neg_pred = torch.cat(hard_neg_pred, 0)
            cl_hard_neg_loss += self.lambda_hnm*self.bceLoss(hard_neg_pred, torch.zeros((hard_neg_pred.shape[0],)).to
            (self.device))
        
        ################### classification loss for regularization ######################
        regu_neg_loss = self.lambda_noobj*self.bceLoss(proposal_map, 
                                     torch.zeros_like(proposal_map).to(self.device))

        ################################## regression ###################################
        reg_loss = self.regressionLoss(torch.tanh(torch.stack(reg_pred, 0)), torch.stack(reg_target, 0))
        loss += cl_pos_loss + cl_neg_loss + cl_hard_neg_loss + regu_neg_loss + reg_loss
        return loss