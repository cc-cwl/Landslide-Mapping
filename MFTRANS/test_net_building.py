import sys
sys.path.insert(0, "Mask2Former")
import tempfile
from pathlib import Path
import numpy as np
import cv2
import os 
#import cog

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config
import torch
import torch.nn as nn
import torch.optim as optim
import time
#from sklearn.metrics import confusion_matrix,cohen_kappa_score
#from sklearn.metrics import f1_score
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
from PIL import Image
__all__ = ['SegmentationMetric']
 
"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""
 
 
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2) # 
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
 
    def meanPixelAccuracy(self):
        #return mean precision for both categories
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
    
    def compute_iou(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        IOU_OUT=IoU[1]
        return IOU_OUT
       
    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        #confusion_matrix = torch.zeros(self.numClass, self.numClass)
        #imgLabel_ts=torch.from_numpy(imgLabel)
        #imgPredict_ts=torch.from_numpy(imgPredict)
        #for t, p in zip(imgLabel_ts.view(-1), imgPredict_ts.view(-1)):
        #        confusion_matrix[t.long(), p.long()] += 1
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        #print(confusionMatrix)
        return confusionMatrix
    
    def compute_recall(self):
        matrix=self.confusionMatrix
        recall=np.diag(matrix)/matrix.sum(axis=0)
        return recall
        
    def compute_f1(self):
        recall=self.compute_recall()
        precision=self.classPixelAccuracy()
        f1=2*recall[1]*precision[1]/(recall[1]+precision[1]+0.0001)
        return f1
        
    def compute_kappa(self):
        matrix=self.confusionMatrix
        pe_rows = np.sum(matrix, axis=0)
        pe_cols = np.sum(matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2+0.0001)
        po = np.trace(matrix) / float(sum_total+0.0001)
        return (po - pe) / (1 - pe+0.0001)
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
    # update confusion matrix
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape # to make sure of same size 
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    # reset confusion matrix
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 
def old():
    imgPredict = np.array([0, 0, 0, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    #print(acc, macc, mIoU)
 
def evaluate1(pre_path, label_path):
    recall_list=[]
    precision_list=[]
    acc_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []
    iou_list = []
    f1_list = []
    kappa_list = []
 
  
    lab_imgs = os.listdir(label_path)
 
    for i, p in enumerate(lab_imgs):
        imgLabel = Image.open(label_path+p)
        imgLabel = np.array(imgLabel)
        
        predict_name=p[:-4]+'_mask2formerv2_5fad.png'
        imgPredict = Image.open(pre_path+predict_name)
        imgPredict = np.array(imgPredict)
        # imgPredict = imgPredict[:,:,0]
 
        metric = SegmentationMetric(2) # 2 is the category number
        metric.addBatch(imgPredict, imgLabel)
        recall=metric.compute_recall()
        precision=metric.classPixelAccuracy()
        acc = metric.pixelAccuracy()
        macc = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
        iou=metric.compute_iou()
        f1=metric.compute_f1()
        kappa=metric.compute_kappa()
 
        recall_list.append(recall)
        precision_list.append(precision)
        acc_list.append(acc)
        macc_list.append(macc)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
        iou_list.append(iou)
        f1_list.append(f1)
        kappa_list.append(kappa)
 
        # print('{}: acc={}, macc={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))
 
    return recall_list, precision_list, acc_list, macc_list, mIoU_list, fwIoU_list, iou_list, f1_list, kappa_list
 
def evaluate2(pre_path, label_path):
    lab_imgs = os.listdir(label_path)
 
    metric = SegmentationMetric(2)  # 2 is the number of category
    for i, p in enumerate(lab_imgs):
        imgLabel = Image.open(label_path+p)
        imgLabel = np.array(imgLabel)
        
        predict_name=p[:-4]+'_mask2formerv2_5fad.png'
        imgPredict = Image.open(pre_path+predict_name)
        imgPredict = np.array(imgPredict)
 
        metric.addBatch(imgPredict, imgLabel)
 
    return metric

def main():
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("./configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml")
    cfg.MODEL.WEIGHTS = '/data1/sgy_mask2former/output/model_0035799.pth'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    #cfg.INPUT.MIN_SIZE_TEST=512
    #cfg.INPUT.MAX_SIZE_TEST=512
    cfg.INPUT.CROP.SIZE = [512,512]
    predictor = DefaultPredictor(cfg)
   
    result_dir = "/data1/sgy_mask2former/data_lushan_cwl/val_result/"
    print("building neural network...")    
    test_image_dir='/data1/sgy_mask2former/data_lushan_cwl/val/JPEGImages/'
    gt_image_dir='/data1/sgy_mask2former/data_lushan_cwl/val/SegmentationClass/'
    
    num_img=len(os.listdir(test_image_dir))
    print(num_img)
    classnum = 2
    time_sum=0
    res=[]
    for index in range(num_img):
       img_name = os.listdir(test_image_dir)[index]
       imgA = cv2.imread('/data1/sgy_mask2former/data_lushan_cwl/val/JPEGImages/'+img_name)
       #v = Visualizer(imgA[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
       start=time.time()
       
       result = predictor(imgA)       
       
       end=time.time()
       res.append(end-start)
       output_array_tensor=result["sem_seg"].argmax(0).to("cpu")
       output_array=output_array_tensor.numpy()
       #semantic_result = v.draw_sem_seg(result["sem_seg"].argmax(0).to("cpu")).get_image()
       #print(semantic_result.shape)
       output_array_save=output_array
       result_path=result_dir+img_name[:-4]+'_mask2formerv2_5fad.png'
       #print(result_path)
       cv2.imwrite(result_path,output_array)
       max_label=np.max(output_array)
    print('finish implementation')
    for i in res:
       time_sum +=i
    print("FPS: %f"%(1.0/(time_sum/len(res))))
    
    # calculate evaluations for each picture and average
    recall_list, precision_list,acc_list, macc_list, mIoU_list, fwIoU_list, iou_list, f1_list, kappa_list = evaluate1(result_dir, gt_image_dir)
    print('final1: recall={:.2f}%, precision={:.2f}%, acc={:.2f}%, macc={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%, iou={:.2f}%, f1={:.2f}%,kappa={:.2f}%'
          .format(np.nanmean(recall_list)*100,np.nanmean(precision_list)*100,np.nanmean(acc_list)*100, np.nanmean(macc_list)*100,
                  np.nanmean(mIoU_list)*100, np.nanmean(fwIoU_list)*100,np.nanmean(iou_list)*100,np.nanmean(f1_list)*100,np.nanmean(kappa_list)*100))
 
    # sum confusion matrix of each picture and calculate evaluations
    metric = evaluate2(result_dir, gt_image_dir)
    recall=metric.compute_recall()
    precision=metric.classPixelAccuracy()
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
    iou=metric.compute_iou()
    f1=metric.compute_f1()
    kappa=metric.compute_kappa()    
 
    print('final2: recall={:.2f}%, precision={:.2f}%, acc={:.2f}%, macc={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%, iou={:.2f}%, f1={:.2f}%, kappa={:.2f}%'
          .format(recall[1]*100, precision[1]*100, acc*100, macc*100, mIoU*100, fwIoU*100,iou*100,f1*100,kappa*100))    
       
    
    


if __name__ == "__main__":
    main()
