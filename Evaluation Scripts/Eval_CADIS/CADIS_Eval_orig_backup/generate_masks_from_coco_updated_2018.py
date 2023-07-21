import json
import os
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import glob
import argparse
import sys

# ht, wt = 960, 1280 #For EV18 and EV17
ht, wt = 540, 960 #For Cadis
ht_start, wt_start = 28, 320
# CATEGORIES = [
#     {"id": 1, "name": "Bipolar Forceps", "supercategory": "Instrument"},
#     {"id": 2, "name": "Prograsp Forceps", "supercategory": "Instrument"},
#     {"id": 3, "name": "Large Needle Driver", "supercategory": "Instrument"},
#     {"id": 4, "name": "Monopolar Curved Scissors", "supercategory": "Instrument"},
#     {"id": 5, "name": "Ultrasound Probe", "supercategory": "Instrument"},
#     {"id": 6, "name": "Suction Instrument", "supercategory": "Instrument"},
#     {"id": 7, "name": "Clip Applier", "supercategory": "Instrument"},
# ]

CATEGORIES = [
    {"id": 1, "name": "Cannula"},
    {"id": 2, "name": "Cap. Cystotome"},
    {"id": 3, "name": "Tissue Forceps"},
    {"id": 4, "name": "Primary Knife"},
    {"id": 5, "name": "Ph. Handpiece"},
    {"id": 6, "name": "Lens Injector"},
    {"id": 7, "name": "I/A Handpiece"},
    {"id": 8, "name": "Secondary Knife"},
    {"id": 9, "name": "Micromanipulator"},
    {"id": 10, "name": "Cap. Forceps"},
]#CHANGED

fold = 'test'
def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data output organization routine MaskRCNN output")
    
    parser.add_argument(
        "--annFile",
        required=False,
        type=str,
        default="/home/arnav/scratch/data/EndoVis_2018/val/coco-annotations/instances_val_sub.json",            #coco-like Ground truth JSON file
        help="annotations file name",
    )
    parser.add_argument(
        "--resFile",
        required=True,
        type=str,
        default = "/home/arnav/scratch/ISINET_evaluation/copy/server/rohan_outputs/maskdino/raw_dumps/test/coco_instances_results.json",  #Non coco-like JSON Predictions file 
        help="Coco instances JSON file from raw dumps",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        default="/home/arnav/scratch/ISINET_evaluation/copy/server/rohan_outputs/maskdino/segms_dumps",
        help="path to the save the masks",
    )
    return parser.parse_args()

def get_filenames(annotations_coco):

    file_names={}
    print(annotations_coco.keys())
    print(annotations_coco["images"])
    for img in annotations_coco["images"]:
        print(img['file_name'])
        file_names[img["id"]]=img['file_name']

    return file_names

def get_imagewise_annotations(data):

    images={  }
    for ann in data:
        if ann["image_id"] not in images.keys():
            images[ann["image_id"]]=[]
        images[ann["image_id"]].append(ann)
    
    return images


instrument_factor = 32
# height, width = 1024, 1280  #For EV18 and EV17
height, width = 540, 960  #For Cadis


args = parse_args()
print("Called with args:")
print(args)
f_ann = open(args.annFile, 'r')  #EndoVis val coco-annotations file (GT)
data_ann = json.load(f_ann)

f_output = open(args.resFile, 'r') #Rohan outputs raw dumps coco-annotations file 
data = json.load(f_output)

segm = []

if not os.path.exists(args.save_dir):
    os.makedirs(os.path.join(args.save_dir))  #Create Save folder as "segms_dumps"

file_names= get_filenames(data_ann)
results = get_imagewise_annotations(data)
print(results.keys())
for id in results.keys():
    mask_instruments = np.zeros((height, width))
    f_name = file_names[int(id)]
    components = f_name.split(".")

    file_name = components[0]

    path_write = args.save_dir   #path_write = ./segms_dumps

    if not os.path.exists(path_write):
        os.makedirs(os.path.join(path_write))

    mask_folder = os.path.join(path_write,fold)  #mask_folder = ./segms_dumps/test 
    if not os.path.exists(mask_folder):
        os.makedirs(os.path.join(mask_folder))

    num_inst = {n: 0 for n in range(1, len(CATEGORIES) + 2)}
    for d in results[id]:
        score = d["score"]
        category_id = d["category_id"]
        image_id = d["image_id"]
        seg = d["segmentation"]
        mask_instruments = np.zeros((height, width))
        mask = maskUtils.decode(seg)
        if mask.sum()> 0:
            if category_id <= 10:  #### Rohan's changes from 7 to 10
                this_inst = num_inst[int(category_id)]
                num_inst[int(category_id)] += 1
            else:
                this_inst = 0
        if category_id == 1:
            mask_instruments[mask > 0] = 1
        elif category_id == 2:
            mask_instruments[mask > 0] = 2
        elif category_id == 3:
            mask_instruments[mask > 0] = 3
        elif category_id == 4:
            mask_instruments[mask > 0] = 4
        elif category_id == 5:
            mask_instruments[mask > 0] = 5
        elif category_id == 6:
            mask_instruments[mask > 0] = 6
        elif category_id == 7:
            mask_instruments[mask > 0] = 7
        elif category_id == 8:
            mask_instruments[mask > 0] = 8
        elif category_id == 9:
            mask_instruments[mask > 0] = 9
        elif category_id == 10:
            mask_instruments[mask > 0] = 10
        elif category_id > 10: #CHANGED
            mask_instruments[mask > 0] = 0
    
        mask_instruments = mask_instruments.astype(np.uint8) * 32
        mask_instruments = cv2.resize(mask_instruments, (960, 540)) #CHANGED DIM,
        path = os.path.join(mask_folder,file_name+"_imageid{}"+"_score{}"+ "_class{}_"+ "inst{}.png").format(image_id, score, category_id, this_inst)
        
        #--------------------------------Creation of segm.json file---------------------------------------------------------
        segmentation = maskUtils.encode(np.asfortranarray(mask_instruments))
        segmentation['counts'] = segmentation['counts'].decode('ascii')
        cat_ids= [category_id]
        # cat_ids+= list(set([1,2,3,4,5,6,7,0]).difference(set([category_id])))
        cat_ids+= list(set([1,2,3,4,5,6,7,8,9,10,0]).difference(set([category_id])))

        ann = {'image_id':image_id, 
		       'category_id':category_id, 
		       'segmentation':segmentation, 
		       'score':score, 
		       'topk_scores':[score,0,0,0,0,0,0,0,0,0,0],   
		       'topk_indices':cat_ids
		}
        #CHANGED PREV TOPK SCORES: 'topk_scores':[score,0,0,0,0,0,0,0]
        segm.append(ann)
    print("Index: ", id)

results_folder = os.path.join(args.save_dir, 'results')
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

if not os.path.exists(os.path.join(results_folder, 'test')):
	os.mkdir(os.path.join(results_folder, 'test'))
	
with open(os.path.join(results_folder, 'test')+"/segm.json",'w') as fn:
	json.dump(segm, fn) 

#Writing the path of the folder for further usage in a temp file.
file1 = open("temp.txt","w")
file1.write(results_folder+'/test'+'/segm.json')
file1.close()
print("Done!\n New File saved at: ", results_folder+'/test'+'/segm.json')    #Converted coco-like Predictions JSON file.
