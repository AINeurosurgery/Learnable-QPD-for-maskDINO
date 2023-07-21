import os
import sys
import glob
from os.path import *
import argparse
#--------------------------------Inputs------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Input data for score calculation")
parser.add_argument(
        "--ground_truth_file",
        required=False,
        type=str, 
        # default = "/home/arnav/scratch/data_britty_mam/2018/val/coco-annotations/instances_val_sub.json" ,     
        default = "/home/arnav/scratch/data_britty_mam/cadis_data/annotations/instances_val_new.json",
        help="annotations file name",
)
parser.add_argument(
        "--prediction_file",
        required=False,
        type=str,
        default = "/home/arnav/scratch/ISINET_evaluation/cleaned_code/output_jsons/cross_check_json/output_after_uploading_github.json",  #Non ISINET-like JSON Predictions file 
        help="Coco instances JSON file from raw dumps",
)
parser.add_argument(
        "--save_dir",
        required=False,
        type=str,
        default="/home/arnav/scratch/ISINET_evaluation/cleaned_code/cross_check/",
        help="path to the save the masks",
)
parser.add_argument(
        "--save_name",
        required=False,
        type=str,
        # default="check.txt",
        default="output_cadis_dummy_after_uploading_to_github.txt",
        help="path to the save the masks",
)
args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------
keyword = 'maskdino'

WORKERS=1
SPLIT='val'
CUDA_VISIBLE_DEVICES=0
# IM_PATH = join(args.ground_truth_file.split('coco-annotations')[0], 'images')  #Cmd for EV18 Dataset
IM_PATH = join(args.ground_truth_file.split('annotations')[0], 'val')  #Cmd for EV17 Dataset
print(IM_PATH)
# BW_PATH = join(args.ground_truth_file.split('coco-annotations')[0], 'annotations') #Cmd for EV18 Dataset
BW_PATH = join(args.ground_truth_file.split('annotations')[0], 'my_annotation_images') #Cmd for EV17 Dataset
WARPED_ANN_DIR=join(args.save_dir, 'test', 'saved_output')
COCO_DIR = args.ground_truth_file
SEGM_DIR = join(args.save_dir, 'test', 'segm.json')

print(SEGM_DIR)
print(dirname(SEGM_DIR))


result_folder = os.path.join(args.save_dir, keyword)
if os.path.isdir(result_folder):
        os.system("rm -rf "+result_folder)
if os.path.isdir(WARPED_ANN_DIR):
        os.system("rm -rf "+WARPED_ANN_DIR)
if not os.path.exists(result_folder):
        os.mkdir(result_folder)
raw_dump_folder = os.path.join(result_folder, 'raw_dumps')
segm_dump_folder = os.path.join(result_folder, 'segms_dumps')
print(segm_dump_folder)
if not os.path.exists(raw_dump_folder):
        os.mkdir(raw_dump_folder)
if not os.path.exists(segm_dump_folder):
        os.mkdir(segm_dump_folder)
        print("Created directory: ", segm_dump_folder)
print(os.path.exists(segm_dump_folder))

folder = "test"
if not os.path.exists(os.path.join(segm_dump_folder, folder)):
        os.mkdir(os.path.join(segm_dump_folder, folder))
if not os.path.exists(os.path.join(raw_dump_folder, folder)):
        os.mkdir(os.path.join(raw_dump_folder, folder))


print("Converting the JSON file ...")
os.system("python generate_masks_from_coco_updated_2018.py "+ "--resFile " + args.prediction_file + " --save_dir " + segm_dump_folder)
print("Converstion complete!")
file1 = open("temp.txt","r+") 
segm_path = (file1.read())
# segm_path = "/home/arnav/scratch/ISINET_evaluation/cleaned_code/maskdino_ablation/cadis/maskdino/segms_dumps/results/test/segm.json"

os.system("python main_authors.py \
        --inference_dataset RobotsegTrackerDataset --inference_dataset_img_dir "+IM_PATH+"\
        --inference_dataset_coco_ann_path "+COCO_DIR+"\
        --inference_dataset_segm_path "+ segm_path+"\
        --inference_dataset_ann_dir " +BW_PATH+ "\
        --inference_dataset_cand_dir " +WARPED_ANN_DIR+ "\
        --inference_dataset_dataset '2018' \
        --inference_dataset_prev_frames 7 \
        --inference_dataset_nms 'True' >> "+args.save_dir+args.save_name)

print("Done! Scores saved at: ", args.save_dir+args.save_name)

os.remove("temp.txt")