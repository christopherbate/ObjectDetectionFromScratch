import torch
import numpy as np
import time
from utils import anchor_box_iou

class DetectionEvaluator(object):
    def __init__(self):
        self.metrics = []
        self.predictions = {}
        self.iou = {}
        self.gt_matches = {}        
        self.dt_matches = {}
        self.iou_thresholds = torch.arange(0.5,0.9,0.1)        
        self.recall_thresholds = torch.arange(0.0,1.0,0.1)
        self.max_detections = 25

    def eval_batch(self, ids, pred_boxes, pred_scores, true_boxes, true_labels):
        '''
        
        '''        
        for idx, imgId in enumerate(ids):            
            if pred_scores[idx].shape[0] == 0:
                continue
            if len(pred_scores[idx].shape) == 1:
                pred_scores[idx].unsqueeze_(dim=-1)

            # Sort the predictions by score.      
            confs, cats = torch.max(pred_scores[idx], dim=-1)            
            sorted_inds = torch.argsort(pred_scores[idx].squeeze(-1), descending=True)[:self.max_detections]
            boxes = pred_boxes[idx][sorted_inds]
            confs = confs[sorted_inds]
            cats = cats[sorted_inds]
            self.predictions[imgId] = {'labels': cats.cpu(), 'boxes': boxes.cpu() ,'confidences': confs.cpu()}
            self.iou[imgId] = anchor_box_iou(true_boxes[idx], boxes).cpu()
            
            # Matches contains array of ground truth matches.
            self.gt_matches[imgId] = torch.zeros((len(self.iou_thresholds), true_boxes[idx].shape[0]), dtype=torch.long)
            self.dt_matches[imgId] = torch.zeros((len(self.iou_thresholds), pred_scores[idx].shape[0]), dtype=torch.long)                        
            
            self.match(imgId)

    def match(self, imgId):
        '''
        Called after eval_batch populated the 
        data for the imgId
        '''
        for iouThreshIdx, iouThresh in enumerate(self.iou_thresholds):
            # Loop over detections - these should be sorted in decreasing confidence
            # in order to give greedy matching priority            
            for dInd in range(self.iou[imgId].shape[1]):
                match = -1
                best_iou = min(iouThresh, 1-1e-10)
                # Loop over ground truth
                for gInd in range(self.iou[imgId].shape[0]):
                    # Skip those already matched. 
                    if self.gt_matches[imgId][iouThreshIdx, gInd] > 0:
                        continue

                    if self.iou[imgId][gInd, dInd] < best_iou:
                        continue

                    best_iou = self.iou[imgId][gInd, dInd]
                    match = gInd

                if match == -1:
                    continue   

                self.gt_matches[imgId][iouThreshIdx, match] = dInd
                self.dt_matches[imgId][iouThreshIdx, dInd] = gInd

    def accumulate(self):
        imgIds = list(self.iou.keys())
        image_metrics = []
        dt_scores = torch.cat([self.predictions[imgId]["confidences"] for imgId in imgIds], dim=0)        
        sortedIdx = torch.argsort(dt_scores, descending=True, dim=0)
        dt_conf_sorted = dt_scores[sortedIdx]
        dt_matches = torch.cat([self.dt_matches[imgId] for imgId in imgIds], dim=1)[:, sortedIdx]
        gt_matches = torch.cat([self.gt_matches[imgId] for imgId in imgIds], dim=1)       
        tp_sum = torch.cumsum(dt_matches.to(torch.bool), dim=-1, dtype=torch.float)
        fp_sum = torch.cumsum(torch.logical_not(dt_matches), dim=-1, dtype=torch.float)            

        '''
        Loop over the iou thresholds and calculate the tp/fp metrics
        '''
        precision = torch.zeros((len(self.iou_thresholds),len(self.recall_thresholds)))
        scores = torch.zeros((len(self.iou_thresholds),len(self.recall_thresholds)))        
        recall = torch.zeros(len(self.iou_thresholds))
        for iouThreshIdx, (tp, fp)  in enumerate(zip(tp_sum, fp_sum)):
            pr = tp / (fp+tp+np.spacing(1))
            rc  = tp / gt_matches.shape[1]    
            for i in range(len(pr)-1, 0, -1):
                if pr[i] > pr[i-1]:          
                    pr[i-1] = pr[i]          
                        
            prec_at_rc = torch.zeros_like(self.recall_thresholds)
            conf_at_rc = torch.zeros_like(self.recall_thresholds)
            inds = np.searchsorted(rc, self.recall_thresholds, side='left')
            for threshIdx, threshInd in enumerate(inds):
                if threshInd < pr.shape[0]:
                    prec_at_rc[threshIdx] = pr[threshInd]
                    conf_at_rc[threshIdx] = dt_conf_sorted[threshInd]

            precision[iouThreshIdx,:] = prec_at_rc
            scores[iouThreshIdx, :] = conf_at_rc
            recall[iouThreshIdx] = rc[-1]            
        self.summarize(precision, scores, recall)
        return precision, scores, recall

    def summarize(self, precision, scores, recall):
        iStr = ' {:<18} @[ IoU={:<9}, Recall={:.2f} ] = {:0.3f}'
        titleStr = 'Average Precision (AP)'        
        iouStr = '{:0.2f}:{:0.2f}'.format(self.iou_thresholds[0], self.iou_thresholds[-1])            
        for idx in range(len(self.recall_thresholds)):            
            print(iStr.format(titleStr, iouStr,self.recall_thresholds[idx], precision[:,idx].mean()))

        titleStr = 'Average Recall (AR)'                
        iStr = ' {:<18} @[ IoU={:<9}] = {:0.3f}'
        print(iStr.format(titleStr, iouStr, recall.mean()))