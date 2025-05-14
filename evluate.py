from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_ap(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_pred, iouType="segm") 
    coco_eval.evaluate()  
    coco_eval.accumulate() 
    coco_eval.summarize()  


# 示例使用
gt_json_path = "/home/wangtuo/workspace/research/SSSI/Data/output2_2.json"  
pred_json_path = "/home/wangtuo/workspace/research/SSSI/Data/text10.json"  

calculate_ap(gt_json_path, pred_json_path)
