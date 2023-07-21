import torch
import torch.nn.functional as F
import torch.utils.data as data

import sys, os, math, random
from os.path import *
import numpy as np

from glob import glob
from tqdm import tqdm

import re
import json
from skimage.transform import resize
import skimage.io as io
import pycocotools.mask as maskUtils


def compute_mask_IU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union
    
class AverageMeter(object):
    """Computes and stores the average and current value."""

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

class RobotsegTrackerDataset(data.Dataset):
    """MICCAI Robotic Scene Segmentation Sub-Challenge dataset."""

    def __init__(self, 
                 img_dir, 
                 ann_dir, 
                 cand_dir,
                 segm_path='./segm.json',
                 coco_ann_path='./coco_ann.json',
                 prev_frames=3,
                 nms=False,
                 maskrcnn_inference=False,
                 dataset='2017'):
        """
        Args:
            img_dir (string): Directory with all the images.
        """
        
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.cand_dir = cand_dir
        self.nms = nms
        if nms:
            self.full_cand_dir = join(cand_dir, 'candidates_nms')
        else:
            self.full_cand_dir = join(cand_dir, 'candidates')

        self.img_list = glob(join(self.img_dir, '*.png'))
        self.img_list.sort()
        self.ann_list = glob(join(self.ann_dir, '*.png'))
        self.ann_list.sort()

        # Load or create
        if not exists(self.full_cand_dir):
            if exists(segm_path):
                if exists(coco_ann_path):
                    if not exists(self.full_cand_dir):
                        os.makedirs(self.full_cand_dir)
                        self.process_candidates(segm_path, coco_ann_path)
                        
                        # Non-maximum supression
                        if nms:
                            cand_info = self.load_json(join(self.cand_dir, 'cand_dataset_nms.json'))
                            img_list = os.listdir(self.img_dir)
                            img_list.sort()
                            cand_list = os.listdir(self.full_cand_dir)
                            for img_name in tqdm(img_list):
                                basename = img_name[:-4]
                                filtered_cands = self.filter_for_candidates(basename, cand_list)
                                for c, cand in enumerate(filtered_cands):
                                    cand_img = io.imread(join(self.full_cand_dir, cand))
                                    overlaped_cands = [cand]
                                    cands_scores = [cand_info[img_name][cand]['scores'][0]]
                                    for other_cand in filtered_cands[c + 1:]:
                                        other_cand_img = io.imread(join(self.full_cand_dir, other_cand))
                                        i, u = compute_mask_IU(cand_img/255, other_cand_img/255)
                                        if i/u > 0.6:
                                            overlaped_cands.append(other_cand)
                                            cands_scores.append(cand_info[img_name][other_cand]['scores'][0])
                                    if len(overlaped_cands) > 1:
                                        cands_scores = np.asarray(cands_scores)
                                        max_score_idx = np.argmax(cands_scores)
                                        overlaped_cands.pop(max_score_idx)
                                        non_maximum_cands = overlaped_cands
                                        for nm_cand in non_maximum_cands:
                                            os.remove(join(self.full_cand_dir, nm_cand))
                                            filtered_cands.remove(nm_cand)
                                            cand_info[img_name].pop(nm_cand)
                                            # tqdm.write('Remove {}'.format(nm_cand))

                            filename = join(self.cand_dir, 'cand_dataset_nms.json')
                            self.save_json(filename, cand_info)
                else:
                    print('No coco annotations and no segmentations found')
                    sys.exit(1)
            else:
               
                print('No segm.json at {} and no candidates found at {}'.format(segm_path, self.feat_dir))
                sys.exit(1)

        if nms:
            self.cand_dataset = self.load_json(join(self.cand_dir, 'cand_dataset_nms.json'))
        else:
            self.cand_dataset = self.load_json(join(self.cand_dir, 'cand_dataset.json'))
        self.cand_dataset_keys = list(self.cand_dataset.keys())
        self.cand_dataset_keys.sort()
        self.prev_frames = prev_frames
        self.frame_list = self.cand_dataset_keys
        
        self.inference_list = torch.ones(len(self.cand_dataset_keys))
        
        seqs = list(set(re.findall('(?<=seq)\d+', ' '.join(self.cand_dataset_keys))))
        #seqs.sort()
       

        for seq in seqs:
            first_frames = self.filter_for_sequence(seq, self.cand_dataset_keys)[:(self.prev_frames - 1)]
            #print("first_frames", first_frames)
            idxs = [self.cand_dataset_keys.index(frame) for frame in first_frames]
            self.inference_list[idxs] = 0
            
        #print("self.cand_dataset_keys", len(self.inference_list), self.inference_list)
        for i, key in enumerate(self.cand_dataset_keys):
            if len(self.cand_dataset[key]) == 0:
                self.inference_list[i] = 0
        maskrcnn_inference = True
        if maskrcnn_inference:
            self.inference_list = torch.zeros(len(self.cand_dataset_keys))


    ## General helper functions
    def load_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, 'w') as f:
            json.dump(data, f)

    def filter_for_annotations(self, img_name, ann_list):
        cated_list = ' '.join(ann_list)
        filtered_anns = re.findall(img_name + "_class\d_inst\d.png", cated_list)
        return filtered_anns

    ## Candidate helper functions
    def process_candidates(self, segm_path, coco_ann_path,
                           min_score=0.6):
        classes = np.zeros([1,10])
        candidates = self.load_json(segm_path)
        img_data = self.load_json(coco_ann_path)
        img_data = img_data['images']
        image_id = 0
        idx = 0
        cand_dataset = {}
        
        for img in tqdm(img_data):
            image_id = img['id']
            name = img['file_name']
            cand = candidates[idx]
            cand_pos = 0
            cand_id = 0
            image_dict = {}

            filtered_anns = self.filter_for_annotations(name[:-4], self.ann_list)
            while cand['image_id'] <= image_id:
                if cand['score'] >= min_score and cand['image_id'] == image_id:
                    cand_name = '{}_{}.png'.format(img['file_name'][:-4], cand_id)
                    # create cand labels
                    segm = cand['segmentation']
                    segm_mask = maskUtils.decode(segm)
                    segm_mask[segm_mask==1] = 255
                    
                    bg_idx = cand['topk_indices'].index(0)
                    cand['topk_indices'].pop(bg_idx)
                    cand['topk_scores'].pop(bg_idx)
                    # create labels
                    cand_labels = [0]*len(cand['topk_indices'])
                    if len(filtered_anns) > 0:
                        best_class = self.get_cand_class(segm_mask, filtered_anns)
                        cand_labels[best_class-1] = 1
                        classes[0,best_class-1] += 1
                    # save candidate
                    cand_name = '{}_{}.png'.format(img['file_name'][:-4], cand_id)
                    io.imsave('{}/{}'.format(join(self.full_cand_dir),
                                             cand_name), segm_mask)
                    # save cand data
                    image_dict[cand_name] = {'classes' : cand['topk_indices'], 
                                            'scores' : cand['topk_scores'],
                                            'labels': cand_labels,
                                            'image_id': image_id,
                                            'cand_feat_pos': cand_pos}
                    cand_id += 1
                cand_pos += 1
                idx += 1
                if idx < len(candidates):
                    cand = candidates[idx]
                else:
                    break
            cand_dataset[img['file_name']] = image_dict
            
        if self.nms:
            self.save_json(join(self.cand_dir, 'cand_dataset_nms.json'), 
                            cand_dataset)
        else:
            self.save_json(join(self.cand_dir, 'cand_dataset.json'), 
                            cand_dataset)

    def load_candidates(self, cand_names):
        candidates = []
        cands = [] # best save instead as a multidim tensor
        for c_name in cand_names:
            filename = join(self.full_cand_dir, c_name)
            if not exists(filename): # special case candidates come from anns
                filename = join(self.bw_ann_dir, c_name) 
            cand = io.imread(filename)
            ####modified for EETS frame 
            #cand = np.array(resize(cand, (1024, 1024)))
            cand = torch.tensor(cand)
            cand = torch.unsqueeze(cand, 0)
            cands.append(cand)
            candidates = torch.cat(cands, 0)
        return candidates

    def filter_for_sequence(self, seq, img_list):
        cated_list = ' '.join(img_list)
        ####modified for EETS frame 
        filtered_frames = re.findall('seq' +seq + "_\d+.png", cated_list)
        return filtered_frames

    def filter_for_candidates(self, img_name, cand_list):
        cated_list = ' '.join(cand_list)
        filtered_cands = re.findall(img_name + "_\d.png", cated_list)
        return filtered_cands

    def get_cand_class(self, cand, ann_names, thresh=0.75): 
        cand = torch.tensor(cand).float()/255
        anns = self.load_annotations(ann_names)/255
        anns = torch.tensor(anns).float()
        num_anns = anns.shape[0]
        ious = torch.zeros(num_anns)
        for idx in range(num_anns):
            i, u = compute_mask_IU(anns[idx,:,:], cand) ## check if done in correct dim
            ious[idx] = i/u
        if torch.max(ious) > thresh:
            idx = torch.argmax(ious).item()
            cand_class = int(re.search(r'\w(?=_inst\d.png)', ann_names[idx]).group())
        else:
            cand_class = 0
        return cand_class

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        org_frame = self.frame_list[idx]
        inference = self.inference_list[idx]
        #print("inference", inference)
        #print("org_frame", org_frame)
        if inference:
            frames = self.frame_list[idx - (self.prev_frames - 1) : idx + 1]

            image_list = []
            for frame1, frame2 in zip(frames[:-1], frames[1:]):
                img1 = io.imread(join(self.img_dir, frame1))
                img2 = io.imread(join(self.img_dir, frame2))
                ####modified for EETS frame 
                img1_resized = np.array(resize(img1, (1024, 1024)))
                img2_resized = np.array(resize(img2, (1024, 1024)))
                images = [img1, img2]
                images = np.array(images).transpose(3,0,1,2)
                images = torch.from_numpy(images.astype(np.float32))
                image_list.append(images)

            image_list.reverse()
                
            candidate_list = []
            preds_list = []
            scores_list = []
            target_list = []
            
            target = torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])
            #print("images",images.shape, len(image_list))
            
            #print("frames", frames) 
            for frame in frames:
                img_data = self.cand_dataset[frame]
                cand_names = list(img_data.keys())
                #print("cand_names", cand_names)
                candidates = self.load_candidates(cand_names)
                #print("candidates", candidates.shape) 
                candidate_list.append(candidates)
                #print("candidate_list", len(candidate_list)) 
                cand_preds = [value['classes'][0] for value in list(img_data.values())]
                cand_scores = [value['scores'][0] for value in list(img_data.values())]
                scores_list.append(cand_scores)
                preds_list.append(cand_preds)
                target_list.append(target)
            #print("preds_list", preds_list)
            target_list = target_list[:-1]
            target_list.reverse()
            candidate_list.reverse()
            preds_list.reverse()   
            scores_list.reverse() 
        else:
            image_list = []
            target_list = []
            img_data = self.cand_dataset[org_frame]
            #print("img_data", img_data)
            cand_names = list(img_data.keys())
            #print("cand_names", cand_names)
            candidate_list = self.load_candidates(cand_names) 
            # print(img_data.values())
            preds_list = [value['classes'][0] for value in list(img_data.values())]
            #print("preds_list", preds_list)
            scores_list = [value['scores'][0] for value in list(img_data.values())]

        full_mask = io.imread(join(self.ann_dir, org_frame))
        # print("FUll mask path", join(self.ann_dir, org_frame))
        full_mask = torch.from_numpy(full_mask)
        return candidate_list, preds_list, scores_list, full_mask