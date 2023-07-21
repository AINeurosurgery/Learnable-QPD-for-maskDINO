import torch
from torch.utils.data import DataLoader
import argparse, os, sys
import numpy as np
from tqdm import tqdm
from glob import glob
import inspect
import time
from inspect import isclass
from os.path import *

import datasets

global param_copy

def module_to_dict(module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

class TimerBlock: 
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

    def log(self, string):
        duration = time.perf_counter() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=list(module_dict.keys()))
    
    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getfullargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)
        if arg not in skip_params + ['self', 'args']:
            if arg in list(parameter_defaults.keys()):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]), default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
            else:
                print(("[Warning]: non-default argument '{}' detected on class '{}'. This argument cannot be modified via the command line"
                        .format(arg, module.__class__.__name__)))

def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in list(vars(args).items()) if argument_for_class in key and key != argument_for_class + 'class'}


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()        
    add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='RobotsegTrackerDataset', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'img_dir': 'images',
                                                        'ann_dir': 'annotations',
                                                        'cand_dir': 'candidates',
                                                        'coco_ann_dir': 'coco_anns.json',
                                                        'segm_dir': 'segm.json',
                                                        'prev_frames': 7,
                                                        'nms': True,
                                                        'maskrcnn_inference': False,
                                                        'dataset': '2017'})

    #Defining the number of classes
    num_classes = 7 #CHANGED
    # num_classes = 10 #CHANGED FOR CADIS
    
    with TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        args.inference_dataset_class = module_to_dict(datasets)[args.inference_dataset]
        args.grads = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference_dataset = args.inference_dataset_class(**kwargs_from_args(args, 'inference_dataset'))
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # Reusable function for inference
    def inference(args, epoch, data_loader, offset=0):  

        progress = tqdm(data_loader, ncols=100, total=len(data_loader) , desc='Calculating scores ', 
            leave=True, position=offset)
        all_im_iou_acc = []
        all_im_iou_acc_challenge = []
        cum_I, cum_U = 0, 0
        class_ious = {c: [] for c in range(1, num_classes+1)}
        changed = 0
        matched = 0
        temp = []
        
        try:
            for batch_idx, (cand_list, pred_list, score_list, full_mask) in enumerate(progress):
            # for batch_idx, (data_list, target_list, cand_list, pred_list, score_list, full_mask, inference, img_name) in enumerate(progress):
                _, h, w = full_mask.size()
                prediction = torch.zeros(((h, w, num_classes + 1)), dtype=torch.float)
                
                outputs = []
                if len(cand_list) > 0:
                    cand_list = cand_list.squeeze(0)

                    for i in range(cand_list.shape[0]):
                        prediction[:, :, pred_list[i]] += cand_list[i, :, :].unsqueeze(2).to(torch.float32) * score_list[i].to(torch.float32)

                prediction = np.argmax(prediction.numpy(), 2) # colapse class dim
                # calculate image challenge metrics
                im_iou = []
                im_iou_challenge = []
                target = full_mask.numpy()
                gt_classes = np.unique(target)
                gt_classes.sort()
                gt_classes = gt_classes[gt_classes > 0] # remove background
                if np.sum(prediction) == 0:
                    if target.sum() > 0: 
                        # Annotation is not empty and there is no prediction
                        all_im_iou_acc.append(0)
                        all_im_iou_acc_challenge.append(0)
                        for class_id in gt_classes:
                            if class_id > 8:   #CLASSES TO BE CHANGED FOR EV18?
                                continue
                            class_ious[class_id].append(0)
                    continue
                
                
                gt_classes = torch.unique(full_mask)
                for class_id in range(1, num_classes + 1): 
                    current_pred = (prediction == class_id).astype(float)
                    current_target = (full_mask.numpy() == class_id).astype(float)
                    if current_pred.astype(float).sum() != 0 or current_target.astype(float).sum() != 0:
                        i, u = compute_mask_IU(current_pred, current_target)       
                        im_iou.append(i/u)
                        cum_I += i
                        cum_U += u
                        class_ious[class_id].append(i/u)
                        if class_id in gt_classes:
                            # consider only classes present in gt
                            im_iou_challenge.append(i/u)
                if len(im_iou) > 0:
                    # to avoid nans by appending empty list
                    all_im_iou_acc.append(np.mean(im_iou))
                if len(im_iou_challenge) > 0:
                    # to avoid nans by appending empty list
                    all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

            # calculate final metrics
            final_im_iou = cum_I / cum_U
            mean_im_iou = np.mean(all_im_iou_acc)
            mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)
            
            final_class_im_iou = torch.zeros(num_classes + 1)
            print('Final cIoU per class:')
            print('| Class | cIoU |')
            print('-----------------')
            for c in range(1, num_classes + 1):
                final_class_im_iou[c-1] = torch.tensor(class_ious[c]).mean()
                print('| {} | {:.5f} |'.format(c, final_class_im_iou[c-1]))
            print('-----------------')
            mean_class_iou = torch.tensor([torch.tensor(values).mean() for c, values in class_ious.items() if len(values) > 0]).mean()
            print('mIoU: {:.5f}, IoU: {:.5f}, challenge IoU: {:.5f}, mean class IoU: {:.5f}'.format(
                                                    final_im_iou,
                                                    mean_im_iou,
                                                    mean_im_iou_challenge,
                                                    mean_class_iou))
            print('Match candidates: {} Changed candidates: {}'.format(matched, changed))
            progress.close()

            return
        except Exception as e:
            print("EXCEPTION OCCURED IN INFERECING AS ", e)

    inference(args=args, epoch=0, data_loader=inference_loader, offset=1)

    print("\n")
    

print("DONE!!")


