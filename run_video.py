import sys
sys.path.insert(0, '/home/vinayak/Downloads/OCL-ReID-CAIR/mmtrack/models/orientation')





import os
import os.path as osp
import numpy as np
import json as js

import tempfile
from argparse import ArgumentParser
import mmcv
import sys
from mmtrack.apis import inference_mot, init_model, inference_sot, extract_feature
from mmtrack.core import imshow_tracks, results2outs
from mmtrack.utils import get_root_logger
from tqdm import main
from multiprocessing import Pool
import threading
import time
from multiprocessing import Process, Manager
import torch
import math
import cv2
import shutil

from utils.visdom import Visdom
from utils import Config
from utils.visualization import Drawer
from Evaluator import Tracker

import fitlog

import random
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.contrib import tzip
import sklearn
import matplotlib.cm as cm
# from utils import 
from mmtrack.models.reid.utils import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS
import json
import time
from tqdm import tqdm
from pathlib import Path


file_path = Path(__file__).resolve()
import sys
sys.path.insert(0, '/home/vinayak/Downloads/OCL-ReID-CAIR/')

def write_to_json(file_path, data):
    # 打开文件读取现有数据，并追加新的数据
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)  # 覆盖写入新内容

def display_webcam_visuals(fps=30):
    """
    Access the webcam and display visuals in real-time.
    
    Args:
        fps (int): Frames per second for display. Default is 30.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        cv2.imshow('Webcam Display', frame)
        
        if cv2.waitKey(int(1.0 / float(fps) * 1000)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def display_webcam_visuals_threaded(fps=30):
    """
    Access the webcam and display visuals in real-time in a separate thread.
    
    Args:
        fps (int): Frames per second for display. Default is 30.
    """
    webcam_thread = threading.Thread(target=display_webcam_visuals, args=(fps,), daemon=True)
    webcam_thread.start()
    return webcam_thread

class TargetIdentificationEvaluator():
    def __init__(self, hyper_config, config, identifier_config):
        self.type = "rpf"
        self.ckpt = None
        self.config = config
        self.hyper_config = hyper_config
        self.identifier_config = identifier_config
        self.identifier_params = Config.fromfile(identifier_config)
        self.tracker = None

        self.input = hyper_config.input
        self.output = hyper_config.output
        self.output_json = hyper_config.output_json
        self.gt_bbox_file = hyper_config.gt_bbox_file
        self.seed = hyper_config.seed
        self.show_result = hyper_config.show_result

        self.fps = 1000

        self.result = {}
    
    def init_work_seed(self, seed=123):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sklearn.random.seed(seed)
        sklearn.utils.check_random_state(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # seed_everything(123)
        # pass

    def init_tracker(self, seed):
        if self.output is not None:
            if not osp.exists(self.output):
                os.makedirs(self.output)
        return Tracker(self.type, self.config, self.ckpt, hyper_config=self.hyper_config, seed=seed, identifier_config=self.identifier_params)

    def run_video(self):
        # load images / video / webcam
        use_webcam = True
        cap = cv2.VideoCapture(0)
        imgs = None
        total_frames = None

        # build the model from a config file and a checkpoint file
        self.init_work_seed(self.seed)
        self.tracker = self.init_tracker(self.seed)

        prog_bar = mmcv.ProgressBar(total_frames) if total_frames is not None else None

        i = 0
        init_bbox = None
        while True:
            if use_webcam:
                ret, frame = cap.read()
                if not ret:
                    break
                img = frame
                img_name = f'{i:06d}.jpg'
            else:
                try:
                    img = imgs[i]
                except Exception:
                    break
                if isinstance(img, str):
                    img_name = os.path.splitext(img)[0]
                    img_path = osp.join(self.input, img)
                    img = mmcv.imread(img_path)
                else:
                    img_name = f'{i:06d}.jpg'

            if i == 0:
                if self.gt_bbox_file is not None:
                    bboxes = mmcv.list_from_file(self.gt_bbox_file)
                    init_bbox = list(map(float, bboxes[0].split(',')))
                else:
                    # use a friendly window name for selectROI
                    init_bbox = list(cv2.selectROI('Select ROI', img, False, False))

                # convert (x1, y1, w, h) to (x1, y1, x2, y2)
                if len(init_bbox) >= 4:
                    init_bbox[2] += init_bbox[0]
                    init_bbox[3] += init_bbox[1]
                else:
                    init_bbox = None
            
            time_start = time.time()
            try:
                result_dict, raw_dict = self.tracker.infer(img, init_bbox, i)
            except IndexError as e:
                # Skip frame if gt_bbox causes indexing issues
                if "too many indices" in str(e):
                    init_bbox = np.array([0, 0, img.shape[1], img.shape[0]])
                    result_dict, raw_dict = self.tracker.infer(img, init_bbox, i)
                else:
                    raise
            time_end = time.time()
            # print('infer time cost {:.3f} s'.format(time_end-time_start))
            # print(raw_dict)
            match_box = result_dict.get("match_box", None)
            target_id = raw_dict.get("target_id", -1)
            target_conf = raw_dict.get("target_conf", -1)
            det_bboxes = raw_dict.get("det_bboxes", None)
            tracks_target_conf_bbox = raw_dict.get("tracks_target_conf_bbox", None)
            threshold = raw_dict.get("threshold", None)
            if target_conf is None:
                target_conf = -1
            # print(threshold)
            
            time_start = time.time()
            if self.output_json is not None:
                self.result[f'{img_name}'] = {}
                self.result[f'{img_name}']['target_info'] = [target_id]+match_box+[target_conf] if match_box is not None else [target_id, 0, 0, 0, 0, target_conf]
                self.result[f'{img_name}']['det_bboxes'] = det_bboxes[0].tolist()
                self.result[f'{img_name}']['tracks_target_conf_bbox'] = tracks_target_conf_bbox
                if threshold is not None:
                    self.result[f'{img_name}']['threshold'] = threshold
                
            if self.show_result or self.output is not None:
                img_disp = img.copy()
                if tracks_target_conf_bbox is not None:
                    for track_id in tracks_target_conf_bbox.keys():
                        _, target_conf, track_bbox = tracks_target_conf_bbox[track_id]
                        if target_conf is None:
                            target_conf = -1
                        cv2.rectangle(img_disp, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 255, 0) if track_id == target_id else (255, 0, 0), 3)
                        cv2.putText(
                            img_disp,
                            f'{track_id}, {target_conf:.2f}',
                            (track_bbox[0]+10, track_bbox[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255) if track_id == target_id else (255, 255, 255),
                            2
                        )
                # cv2.putText(
                #         img_disp,
                #         f'Frame: {i:05d}',
                #         (10, 30),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         1,
                #         (255, 255, 255),
                #         3
                #     )
                # # Add threshold to the top right corner
                # if threshold is not None:
                #     cv2.putText(
                #         img_disp,
                #         f'Threshold: {threshold:.2f}',
                #         (img_disp.shape[1] - 350, 30),  # Adjust the position as needed
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         1,
                #         (255, 255, 255),
                #         3
                #     )
            
            # show tracking result
            if self.show_result:
                cv2.imshow('Tracking Frame', img_disp)
                # determine a safe display FPS
                display_fps = getattr(self, 'fps', None)
                if use_webcam and cap is not None:
                    try:
                        cap_fps = cap.get(cv2.CAP_PROP_FPS)
                        if cap_fps and cap_fps > 0:
                            display_fps = cap_fps
                    except Exception:
                        pass
                if not display_fps or display_fps <= 0:
                    display_fps = 30
                if cv2.waitKey(int(1.0 / float(display_fps) * 1000)) & 0xFF == ord('q'):
                    break

            if self.output is not None:
                cv2.imwrite(os.path.join(self.output, f'{i:06d}.jpg'), img_disp)
            time_end = time.time()
            # print('image time cost {:.3f} s'.format(time_end-time_start))
            if prog_bar is not None:
                prog_bar.update()
            i += 1
        if self.output_json is not None:
            write_to_json(self.output_json, self.result)
        if use_webcam and cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Starting webcam visuals...~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    sys.path.insert(0, '/home/vinayak/Downloads/OCL-ReID-CAIR/')
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input video or image directory', default=None)
    parser.add_argument('--output', type=str, default=None, help='path to save the output images')
    parser.add_argument('--show_result', action='store_true', help='whether to display the tracking result')
    parser.add_argument('--method', type=str, choices=['part-OCLReID', 'global-OCLReID', 'rpf-ReID'], default='part-OCLReID', help='tracking method')
    parser.add_argument('--img_width', type=int, default=512, help='image width')
    parser.add_argument('--img_height', type=int, default=384, help='image width')
    parser.add_argument('--output_json', type=str, default=None, help='path to save the output tracking json')

    args = parser.parse_args()

    method = args.method


    base_dir = osp.join(file_path.parent, "tpt_configs")
    to_be_runned = {"rpf-ReID": "baseline_oclreid_resnet18.py",
                    "global-OCLReID": "baseline_oclreid_finenued_global_resnet18.py",
                    "part-OCLReID": "baseline_oclreid_finenued_resnet18.py",}

    print("\nRunning: {}\n".format(method))
    hyper_params = Config.fromfile(osp.join(base_dir, to_be_runned[method]))

    hyper_params.mmtracking_dir = file_path.parent
    if args.input is None:
        args.input = osp.join(file_path.parent, "demo_video.mp4")
    hyper_params.input = args.input
    hyper_params.output = args.output
    hyper_params.show_result = args.show_result
    hyper_params.image_shape = (args.img_width, args.img_height, 3)
    hyper_params.output_json = args.output_json

    print(hyper_params)

    rpf_config = osp.join(hyper_params.mmtracking_dir, hyper_params.rpf_config)
    rpf_config = mmcv.Config.fromfile(rpf_config)
    # load reid ckpt path
    rpf_config.model.reid.init_cfg.checkpoint = osp.join(file_path.parent, "checkpoints/reid/resnet18.pth")

    identifier_config = osp.join(hyper_params.mmtracking_dir, hyper_params.identifier_config)

    evaluator = TargetIdentificationEvaluator(hyper_params, rpf_config, identifier_config)
    evaluator.run_video()

    
