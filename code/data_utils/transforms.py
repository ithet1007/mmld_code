import torch
import numpy as np
from scipy.ndimage.interpolation import zoom


def zoomout_imgandlandmark(img, landmarks, rate):
    new_img = zoom(img, rate, order=1)
    new_landmarks = []
    for position in landmarks:
        position_c = position[0] * rate[0]
        position_h = position[1] * rate[1]
        position_w = position[2] * rate[2]
        new_landmarks.append(np.array([position_c, position_h, position_w]))
    return new_img, np.array(new_landmarks)


class RandomCrop(object):
    def __init__(self, min_rate=0.6, size=[128,128,64]):
        self.size = np.array(size)
        self.min_rate = min_rate

    def __call__(self, sample):
        img = sample['image']
        landmarks = sample['landmarks']
        min_ = np.ones((3,)) * 1000
        max_ = np.zeros((3,))
        for landmark in landmarks:
            for i in range(3):
                # we use a very small value to indicate nonexist landmark
                if np.mean(landmark)< -100:
                    continue
                if min_[i] > landmark[i]:
                    min_[i] = landmark[i]
                if max_[i] < landmark[i]:
                    max_[i] = landmark[i]
        
        # according to the min and max of landmarks to set the maximum zoom rate
        zoom_max = [self.size[0]/(max_[0]-min_[0])-0.02, self.size[1]/(max_[1]-min_[1])-0.02, self.size[2]/(max_[2]-min_[2])-0.04]

        ######################### zoom out #############################
        random_rate0 = np.random.uniform(self.min_rate, min(zoom_max[0], 1)) 
        random_rate1 = np.random.uniform(self.min_rate, min(zoom_max[1], 1))
        random_rate2 = np.random.uniform(self.min_rate, min(zoom_max[2], 1))
        if zoom_max[0]<self.min_rate or zoom_max[1]<self.min_rate or zoom_max[2]<self.min_rate:
            random_rate0=zoom_max[0]
            random_rate1=zoom_max[1]
            random_rate2=zoom_max[2]
        img, landmarks = zoomout_imgandlandmark(img, landmarks, [random_rate0,random_rate1,random_rate2])

        min_ = np.ones((3,)) * 1000
        max_ = np.zeros((3,))
        for landmark in landmarks:
            for i in range(3):
                if np.mean(landmark)< -100:
                    continue
                if min_[i] > landmark[i]:
                    min_[i] = landmark[i]
                if max_[i] < landmark[i]:
                    max_[i] = landmark[i]
        ######################### cropping ###############################
        begin_=(min_+max_)/2.-self.size/2.
        bc = max(0, begin_[0]); ec = min(min_[0], img.shape[0]-self.size[0])
        bh = max(0, begin_[1]); eh = min(min_[1], img.shape[1]-self.size[1])
        bw = max(0, begin_[2]); ew = min(min_[2], img.shape[2]-self.size[2])
        if ec - bc < 1:
            ec += 1
        if eh - bh < 1:
            eh += 1
        if ew - bw < 1:
            ew += 1
        cc = np.random.randint(bc, ec)
        ch = np.random.randint(bh, eh)
        cw = np.random.randint(bw, ew)
        # random crop here
        cur_crop_img = img[cc:(cc+self.size[0]), ch:(ch+self.size[1]), cw:(cw+self.size[2])]
        
        if(cur_crop_img.shape[0]!=self.size[0] or cur_crop_img.shape[1]!=self.size[1] or cur_crop_img.shape[2]!=self.size[2]):
            print(cc, ch, cw)
            print(img.shape)
            print(cur_crop_img.shape)
            print('get a error crop img')
        pre_new_landmarks = []
        for landmark in landmarks:
            cur_landmark = landmark - np.array([cc, ch, cw])          
            pre_new_landmarks.append(cur_landmark)
        sample['landmarks'] = np.array(pre_new_landmarks) 
        sample['image'] = cur_crop_img  
        return sample

    
class LandmarkProposal(object):
    def __init__(self, size=[128,128,64], shrink=4., anchors=[0.5, 0.75, 1., 1.25], max_num=400):
        self.size = size 
        self.shrink = shrink
        self.anchors = anchors
        self.max_num = max_num # setting a fixed anchor number for minibatch

    def __call__(self, sample):
        landmarks = sample['landmarks']
        landmarks = landmarks / self.shrink # shrinking the landmark coordinates
        proposals = []

        for idx, anchor in enumerate(self.anchors):
            proposal = []
            for ldx, landmark in enumerate(landmarks):
                if np.mean(landmark) < -100:
                    cur_ldx = -1 - ldx # negative number indicates nonexist landmarks
                    proposal.append([0,0,0,0,0,0,cur_ldx]) 
                    continue
                else:
                    cur_ldx = ldx

                # if a landmark exist, calculate the proposals
                cl_min = landmark - anchor
                cl_max = landmark + anchor
                c = max(0, int(cl_min[0]))
                max_c = int(np.ceil(cl_max[0])); max_w = int(np.ceil(cl_max[1])); max_h = int(np.ceil(cl_max[2]))
                while(c<=max_c and c<self.size[0]/self.shrink):
                    cur_c = c
                    c += 1
                    w = max(0, int(cl_min[1]))
                    while(w<=max_w and w<self.size[1]/self.shrink):
                        cur_w = w
                        w += 1
                        h = max(0, int(cl_min[2]))
                        while(h<=max_h and h<self.size[2]/self.shrink):
                            cur_h = h
                            h+= 1
                            pred_c = landmark[0]-cur_c; pred_w = landmark[1]-cur_w; pred_h = landmark[2]-cur_h
                            if(np.abs(pred_c)<anchor and np.abs(pred_w)<anchor and np.abs(pred_h)<anchor):
                                proposal.append([cur_c,cur_w,cur_h,pred_c/anchor, pred_w/anchor, pred_h/anchor, cur_ldx])
            
            # if getting too many proposals, truncating it
            if(len(proposal)>=self.max_num):
                print("too many proposals were found !")
                proposal = proposal[:self.max_num]
            # if getting less proposals, padding the tensor
            if len(proposal)<self.max_num:
                proposal += [[0,0,0,0,0,0,-100]] * (self.max_num-len(proposal)) # -100 indicates the padding proposals
            proposals.append(np.array(proposal))
        sample['proposals'] = np.stack(proposals, 0).astype('float32')
        return sample


class CenterCrop(object):
    def __init__(self, size=[128,128,64]):
        self.size = np.array(size)

    def __call__(self, sample):
        img = sample['image']
        landmarks = sample['landmarks']

        min_ = np.ones((3,)) * 1000
        max_ = np.zeros((3,))
        for landmark in landmarks:
            for i in range(3):
                if np.mean(landmark)< -100:
                    continue
                if min_[i] > landmark[i]:
                    min_[i] = landmark[i]
                if max_[i] < landmark[i]:
                    max_[i] = landmark[i]
        zoom_max = [self.size[0]/(max_[0]-min_[0])-0.02, self.size[1]/(max_[1]-min_[1])-0.02, self.size[2]/(max_[2]-min_[2])-0.04]

        ######################### zoom out #############################
        random_rate0 = min(zoom_max[0], 1) 
        random_rate1 = min(zoom_max[1], 1)
        random_rate2 = min(zoom_max[2], 1)
        img, landmarks = zoomout_imgandlandmark(img, landmarks, [random_rate0,random_rate1,random_rate2])

        min_ = np.ones((3,)) * 1000
        max_ = np.zeros((3,))
        for landmark in landmarks:
            for i in range(3):
                if np.mean(landmark)< -100:
                    continue
                if min_[i] > landmark[i]:
                    min_[i] = landmark[i]
                if max_[i] < landmark[i]:
                    max_[i] = landmark[i]
        # import pdb; pdb.set_trace()
        begin = ((max_ + min_) /2 - self.size/2 ).astype("int32")
        begin[0] = max(0, min(begin[0], img.shape[0]-self.size[0]) )
        begin[1] = max(0, min(begin[1], img.shape[1]-self.size[1]))
        begin[2] = max(0, min(begin[2], img.shape[2]-self.size[2]))

        if begin[0]+self.size[0] > img.shape[0] or begin[1]+self.size[1] > img.shape[1] or begin[2]+self.size[2] > img.shape[2]:
            print("find a very small landmark , error !!!!!")
        # center crop here
        sample["image"] = img[begin[0]:begin[0]+self.size[0], begin[1]:begin[1]+self.size[1], begin[2]:begin[2]+self.size[2]]
        landmarks[:, 0] = landmarks[:, 0] - begin[0]
        landmarks[:, 1] = landmarks[:, 1] - begin[1]
        landmarks[:, 2] = landmarks[:, 2] - begin[2]
        sample["landmarks"] = landmarks
        return sample


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        img /= 255.0
        sample['image'] = img
        return sample

    
class LandMarkToGaussianHeatMap(object):
    def __init__(self, R=20., img_size=(128,128,64), n_class=14, GPU=None):
        self.R = R # gaussian heatmap radius
        self.GPU = GPU

        # generate index in three views: length, width, height 
        c_row = np.array([i for i in range(img_size[0])])
        c_matrix = np.stack([c_row] * img_size[1], 1)
        c_matrix = np.stack([c_matrix] * img_size[2], 2)
        c_matrix = np.stack([c_matrix] * n_class, 0)
        
        h_row = np.array([i for i in range(img_size[1])])
        h_matrix = np.stack([h_row] * img_size[0], 0)
        h_matrix = np.stack([h_matrix] * img_size[2], 2)
        h_matrix = np.stack([h_matrix] * n_class, 0)

        w_row = np.array([i for i in range(img_size[2])])
        w_matrix = np.stack([w_row] * img_size[0], 0)
        w_matrix = np.stack([w_matrix] * img_size[1], 1)
        w_matrix = np.stack([w_matrix] * n_class, 0)
        if GPU is not None:
            self.c_matrix = torch.tensor(c_matrix).float().to(self.GPU)
            self.h_matrix = torch.tensor(h_matrix).float().to(self.GPU)
            self.w_matrix = torch.tensor(w_matrix).float().to(self.GPU)

    def __call__(self, landmarks):
        n_landmark = landmarks.shape[1]
        batch_size = landmarks.shape[0]

        if self.GPU is not None:
            # generate the mask inside the mask with radius R
            mask = torch.sqrt(
                torch.pow(
                    self.c_matrix -
                    torch.tensor(np.expand_dims(np.expand_dims(landmarks[:, :, 0:1], 3),4)).float().to(self.GPU), 2) + torch.pow(
                    self.h_matrix -
                    torch.tensor(np.expand_dims(np.expand_dims(landmarks[:, :, 1:2], 3),4)).float(
                    ).to(self.GPU), 2) + torch.pow(
                        self.w_matrix -
                        torch.tensor(np.expand_dims(np.expand_dims(landmarks[:, :, 2:3], 3),4)
                                     ).float().to(self.GPU), 2)) <= self.R

            # generate the heatmap with Gaussian distribution
            # the maximum value is 2, the min value is -1
            cur_heatmap = torch.exp(-((
                torch.pow(
                self.c_matrix - torch.tensor(
                    np.expand_dims(np.expand_dims(landmarks[:, :, 0:1], 3),4)).float().to(
                        self.GPU), 2) + torch.pow(
                self.h_matrix - torch.tensor(
                    np.expand_dims(np.expand_dims(landmarks[:, :, 1:2], 3),4)).float().to(
                        self.GPU), 2) + torch.pow(
                            self.w_matrix - torch.tensor(
                                np.expand_dims(np.expand_dims(landmarks[:, :, 2:3],
                                               3),4)).float().to(self.GPU), 2)) /
                                      (self.R * self.R) / 0.2)) 
            heatmap = 2 * cur_heatmap * mask.float() + mask.float() - 1    
        return heatmap

    
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        img = np.expand_dims(img, 0)
        sample['image'] = img
        sample['landmarks'] = sample['landmarks'].astype(np.float32)
        return sample
    
    
    