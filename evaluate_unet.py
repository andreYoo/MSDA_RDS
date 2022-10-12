import pdb
from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-ground_truth_dir', type=str, default='./image/annot', help='path where ground truth images are located')
    arg('-pred_dir', type=str, default='./r3_pred',  help='path with predictions')
    arg('-threshold', type=float, default=0.1, required=False,  help='crack threshold detection')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    rf = 0
    paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    total_t = 0
    total_tp = 0
    for file_name in tqdm(paths):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)
        _tmp_name = file_name.name.split('.')[0]+'.jpg'
        _img_name = './image/img/'+file_name.name.split('.')[0][0:len(file_name.name.split('.')[0])-5]+'img.png'
        pred_file_name = Path(args.pred_dir) / _tmp_name
        if not pred_file_name.exists():
            print(f'missing prediction for file {file_name.name}')
            continue
        pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)
        img  = cv2.imread(str(_img_name))
        y_pred = pred_image
        total_t += np.sum(y_true==1)
        total_tp += np.sum((y_true+y_pred)==2)
        plt.subplot(141)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(y_true)
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(y_pred)
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(y_true)
        plt.axis('off')
        plt.imshow(y_pred, alpha=0.5)

        plt.savefig('./compare/%d.jpg'%(rf), dpi=500)
        rf +=1
        plt.close('all')
        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]

    print('True positive rate (TP/T) : %f'%(total_tp/total_t))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))