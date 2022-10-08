import logging
import time
import tools._init_path
import os
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg, cfg_from_file, save_config_to_file
import mindspore as ms
from mindspore.context import set_context,PYNATIVE_MODE
set_context(mode=PYNATIVE_MODE)
from pathlib import Path
# from KITTIcols import colums
import numpy as np

# train_mode = 'rpn'
# cfg_file = "cfgs/default.yaml"

# if cfg_file is not None:
#     cfg_from_file(cfg_file)
#     assert not cfg.RPN.USE_INTENSITY
#     tag = os.path.splitext(os.path.basename(cfg_file))[0]

#     if train_mode == 'rpn':
#         cfg.RPN.ENABLED = True
#         cfg.RCNN.ENABLED = False
#         root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
#     elif train_mode == 'rcnn':
#         cfg.RCNN.ENABLED = True
#         cfg.RPN.ENABLED = cfg.RPN.FIXED = True
#         root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
#     elif train_mode == 'rcnn_offline':
#         cfg.RCNN.ENABLED = True
#         cfg.RPN.ENABLED = False
#         root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
#     else:
#         raise NotImplementedError

class batchpad():
    def __init__(self,cols):
        self.cols = cols
    
    def __call__(self,*c):
        # c = list(c)
        ans = []
        BatchInfo = c[-1]
        assert len(self.cols) == len(c)-1
        batch_size = len(c[0])
        # assert batch_size > 1
        for ii,key in enumerate(self.cols):
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 1
                for k in range(batch_size):
                    max_gt = max(max_gt, c[ii][k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :c[ii][i].__len__(), :] = c[ii][i]
                # c[ii] = batch_gt_boxes3d
                ans.append(batch_gt_boxes3d)
                continue
            if isinstance(c[ii], np.ndarray):
                if batch_size == 1:
                    ans.append(c[ii][np.newaxis, ...])
                    # c[ii] = c[ii][np.newaxis, ...]
                else:
                    ans.append(np.concatenate([c[ii][k][np.newaxis, ...] for k in range(batch_size)], axis=0))
                    # c[ii] = np.concatenate([c[ii][k][np.newaxis, ...] for k in range(batch_size)], axis=0)
            else:
                temp = [c[ii][k] for k in range(batch_size)]
                if isinstance(c[ii][0], int):
                    temp = np.array(temp, dtype=np.int32)
                elif isinstance(c[ii][0], float):
                    temp = np.array(temp, dtype=np.float32)
                ans.append(temp)
                # c[ii] = temp
        return tuple(ans)
    

def get_cols(mode="TRAIN"):
    # if cfg.RPN.ENABLED:
    #     return self.get_rpn_sample(index)
    # elif cfg.RCNN.ENABLED:
    #     if self.mode == 'TRAIN':
    #         if cfg.RCNN.ROI_SAMPLE_JIT:
    #             return self.get_rcnn_sample_jit(index)
    #         else:
    #             return self.get_rcnn_training_sample_batch(index)
    #     else:
    #         return self.get_proposal_from_file(index)
    # else:
    #     raise NotImplementedError
    if cfg.RPN.ENABLED:
        return [
            "sample_id", "pts_input", "pts_rect", "pts_features",
            "rpn_cls_label", "rpn_reg_label", "gt_boxes3d"
        ]
    elif cfg.RCNN.ENABLED:
        if mode == 'TRAIN':
            if cfg.RCNN.ROI_SAMPLE_JIT:
                return [
                    'sample_id', 'rpn_xyz', 'rpn_features', 'rpn_intensity',
                    'seg_mask', 'roi_boxes3d', 'gt_boxes3d', 'pts_depth'
                ]
            else:
                # return [
                #     'sample_id', 'pts_input', 'pts_features', 'cls_label',
                #     'reg_valid_mask', 'gt_boxes3d_ct', 'roi_boxes3d',
                #     'roi_size', 'gt_boxes3d'
                # ]
                return [
                    'sample_id', 'pts_input', 'pts_features', 'cls_label',
                    'reg_valid_mask', 'gt_boxes3d_ct', 'roi_boxes3d',
                    'roi_size'
                ]
        else:
            return [
                'sample_id', 'pts_input', 'pts_features', 'roi_boxes3d',
                'roi_scores', 'roi_size', 'gt_boxes3d', 'gt_iou'
            ]
    else:
        raise NotImplementedError

def create_dataloader(logger,args):
    #DATA_PATH = os.path.join('../', 'data')
    DATA_PATH = (Path(__file__).parent.parent.absolute()/'data').absolute()
    # cols = get_cols("TRAIN")
    # create dataloader
    
    train_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.SPLIT, mode='TRAIN',
                                 logger=logger,
                                 classes=cfg.CLASSES,
                                 rcnn_training_roi_dir=args.rcnn_eval_roi_dir,
                                 rcnn_training_feature_dir=args.rcnn_eval_feature_dir,
                                 gt_database_dir=args.gt_database)

    num_class = train_set.num_class
    cols = train_set.getitem_cols(0)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
    #                           num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch,
    #                           drop_last=True)
    # cols = ["sample_id","pts_input","pts_rect","pts_features","rpn_cls_label","rpn_reg_label","gt_boxes3d"]
    train_loader = ms.dataset.GeneratorDataset(train_set,num_parallel_workers=1,column_names=cols,shuffle=True)
    # train_loader.set_dynamic_columns(columns=colums) 
    train_batch_loader = train_loader.batch(args.batch_size,drop_remainder=True,num_parallel_workers=4,python_multiprocessing=True)
    
    if args.train_with_eval:
        test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.VAL_SPLIT, mode='EVAL',
                                    logger=logger,
                                    classes=cfg.CLASSES,
                                    rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                    rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True,num_workers=args.workers, collate_fn=test_set.collate_batch)
        test_loader = ms.dataset.GeneratorDataset(test_set,num_parallel_workers=args.workers,shuffle=True)
        test_loader = test_loader.batch(1,drop_remainder=True,num_parallel_workers=4)
    else:
        test_loader = None
    # return train_set
    return train_batch_loader, test_loader,num_class



def create_dataloader_debug(logger):
    #DATA_PATH = os.path.join('../', 'data')
    DATA_PATH = (Path(__file__).absolute().parent.parent.absolute()/'data').absolute()
    # cols = get_cols("TRAIN")
    # create dataloader
    
    train_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.SPLIT, mode='TRAIN',
                                 logger=logger,
                                 classes=cfg.CLASSES,
                                 rcnn_training_roi_dir=None,
                                 rcnn_training_feature_dir=None,
                                 gt_database_dir='tools/gt_database/train_gt_database_3level_Car.pkl')
    num_class = train_set.num_class
    # train_set.getitem_cols(0)
    # print(train_set[0][2])
    # t1 = time.perf_counter()
    # print()
    # print(time.perf_counter()-t1)
    # exit()
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
    #                           num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch,
    #                           drop_last=True)
    # cols = ["sample_id","pts_input","pts_rect","pts_features","rpn_cls_label","rpn_reg_label","gt_boxes3d"]
    cols = train_set.getitem_cols(0)
    print(cols)
    train_loader = ms.dataset.GeneratorDataset(train_set,num_parallel_workers=1,column_names=cols,shuffle=True)
    # train_loader.set_dynamic_columns(columns=colums) 
    train_batch_loader = train_loader.batch(8,drop_remainder=True,num_parallel_workers=4,per_batch_map=batchpad(cols=cols),python_multiprocessing=False)
    # train_batch_loader = train_loader.batch(1,drop_remainder=True,num_parallel_workers=4,python_multiprocessing=True)


    if False:
        test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.VAL_SPLIT, mode='EVAL',
                                    logger=logger,
                                    classes=cfg.CLASSES,
                                    rcnn_eval_roi_dir=None,
                                    rcnn_eval_feature_dir=None)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True,num_workers=args.workers, collate_fn=test_set.collate_batch)
        test_loader = ms.dataset.GeneratorDataset(test_set,num_parallel_workers=args.workers,shuffle=True)
        test_loader = test_loader.batch(1,drop_remainder=True,num_parallel_workers=args.workers)
    else:
        test_loader = None
    # return train_set
    return train_batch_loader, test_loader,num_class

def demo():
    t,_,n = create_dataloader_debug(logger=logging.getLogger())
    di = t.create_dict_iterator()
    for task in di:
        for k,v in task.items():
            print(k,v.shape)
        exit()

if __name__ == "__main__":
    demo()
    # test passed