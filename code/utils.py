import os
import torch
import numpy as np

def setgpu(gpus):
    if gpus=='all':
        gpus = '0,1,2,3'
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def metric(heatmap, spacing, landmarks):
    N = heatmap.shape[0]
    n_class = heatmap.shape[1]
    total_mre = []
    max_num = 500
    hits = np.zeros((8, n_class))

    for j in range(N):
        cur_mre_group = []
        for i in range(n_class):
            max_count = 0
            group_rate = 0.999
            if np.max(heatmap[j,i])>0:
                while max_count < max_num:
                    h_score_idxs = np.where(
                        heatmap[j, i] >= np.max(heatmap[j, i])*group_rate)
                    group_rate = group_rate - 0.1
                    max_count = len(h_score_idxs[0])
            else:
                h_score_idxs = np.where(
                    heatmap[j, i] >= np.max(heatmap[j, i])*(1+0.5))
            
            h_predict_location = np.array(
                [np.mean(h_score_idxs[0]), np.mean(h_score_idxs[1]), np.mean(h_score_idxs[2])])

            cur_mre = np.linalg.norm(
                np.array(landmarks[j,i] - h_predict_location)*spacing, ord=2)

            if np.mean(landmarks[j, i])>0:
                cur_mre_group.append(cur_mre) 
                hits[4:, i] += 1
                if cur_mre <= 2.0:
                    hits[0, i] += 1
                if cur_mre <= 2.5:
                    hits[1, i] += 1
                if cur_mre <= 3.:
                    hits[2, i] += 1
                if cur_mre <= 4.:
                    hits[3, i] += 1
            else:
                cur_mre_group.append(-1) 
        total_mre.append(np.array(cur_mre_group))

    return total_mre, hits
    

def min_distance_voting(landmarks):
    min_dis = 1000000
    min_landmark = landmarks[0]
    for landmark in landmarks:
        cur_dis = 0
        for sub_landmark in landmarks:
            cur_dis += np.linalg.norm(
            np.array(landmark - sub_landmark), ord=2)
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_landmark = landmark
    return min_landmark


def metric_proposal(proposal_map, spacing, 
                     landmarks, shrink=4., anchors=[0.5, 1, 1.5, 2], n_class=14):
    # selected number for candidate landmark voting for one landmark
    # can be fine-tuned according to anchor numbers
    select_number = 15 
    
    batch_size = proposal_map.size(0)
    c = proposal_map.size(1)
    w = proposal_map.size(2)
    h = proposal_map.size(3)
    n_anchor = proposal_map.size(4)
    total_mre = []
    hits = np.zeros((8, n_class))
    
    for j in range(batch_size):
        cur_mre_group = []
        for idx in range(n_class):
            #################### from proposal map to landmarks #########################
            proposal_map_vector = proposal_map[:,:,:,:,:,3+idx].reshape(-1)
            mask = torch.zeros_like(proposal_map_vector)
            _, cur_idx = torch.topk(
                        proposal_map_vector, select_number)
            mask[cur_idx] = 1
            mask_tensor = mask.reshape((batch_size, c, w, h, n_anchor, -1))
            select_index = np.where(mask_tensor.cpu().numpy()==1)
               
            # get predicted position
            pred_pos = []
            for i in range(len(select_index[0])):
                cur_pos = []
                cur_batch = select_index[0][i] 
                cur_c = select_index[1][i]
                cur_w = select_index[2][i]
                cur_h = select_index[3][i]
                cur_anchor = select_index[4][i]
                cur_predict = torch.tanh(proposal_map[cur_batch, cur_c, cur_w, cur_h, cur_anchor, :3]).cpu().numpy()

                cur_pos.append( (np.array([cur_c, cur_w, cur_h]) + cur_predict*anchors[cur_anchor])*shrink )
                pred_pos.append(cur_pos)
            pred_pos = np.array(pred_pos)
 
            cur_mre = np.linalg.norm(
                (np.array(landmarks[j,idx] - min_distance_voting(pred_pos)))*spacing[j], ord=2)
            if cur_mre <= 2.0:
                hits[0, idx] += 1
            if cur_mre <= 2.5:
                hits[1, idx] += 1
            if cur_mre <= 3.:
                hits[2, idx] += 1
            if cur_mre <= 4.:
                hits[3, idx] += 1
                
            if np.mean(landmarks[j, idx])>0:
                cur_mre_group.append(cur_mre) 
                hits[4:, idx] += 1
            else:
                # if landmark nonexist, do not calculate MRE and SDR, using -1 to indicate it
                cur_mre_group.append(-1) 
        total_mre.append(np.array(cur_mre_group))
        return total_mre, hits
                