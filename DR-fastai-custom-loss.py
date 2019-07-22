#!/usr/bin/env python
# coding: utf-8
# +
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import scipy as sp
from functools import partial
from sklearn.metrics import cohen_kappa_score

# for image preprocessing
import cv2
#import imutils
from torchvision.transforms import ToPILImage
from shutil import copyfile

import torch
from fastai import *
from fastai.core import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.torch_core import *
from fastai.callbacks import CSVLogger

from fastai.vision import *
from fastai.vision.learner import create_head, cnn_config, num_features_model

from IPython.core.debugger import set_trace
# -

# +
# this cell is for fast submitting, refer https://www.kaggle.com/c/instant-gratification/discussion/94379#latest-546086
# if you are sure your code is right, keep `fast_commit` to True, so after running this cell, code commission can be
# done and you can submit the kernel for LB quickly.

# but it is recommended to first set `fast_commit` to false first and the kernel will run with only partial data to
# check if there is bug in the code.
fast_commit = True  # commit just to check run OK, we can use this for debugging. Write unittest suit in this jupyter notebook
fast_commit_with_commit_runing_less_data = True  # otherwise just exit

random_world = True
final_submission = False  # for disable random seed setting. Also, we can use all training data (no validation) for
                          # final submission
do_lr_find = False
try:
    sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
except:
    sub = pd.read_csv('../input/sample_submission.csv')

submitting_to_LB = False
use_less_train_data = False

if fast_commit:
    if len(sub) < 2000:  # commit, not submit to leaderboard
        sub.to_csv('submission.csv', index=False)  # so we can always submit
        if fast_commit_with_commit_runing_less_data:
            use_less_train_data = True
        else:
            exit()
    else:  # this is real submit for leader board
        submitting_to_LB = True
else:  # pretending/testing for submitting to LB
    submitting_to_LB = True
    #do_lr_find = True
# -

# +
# %reload_ext autoreload
# %autoreload 2
# #!nvidia-smi
# -


# ## Utils functions
# helpful functions, for debugging mainly


# +
# not used...
def binary_search_img(start, end, f, steps_to_stop):  # we could use the contour information!!
    if start >= end: return start
    if steps_to_stop <= 0:
        raise RuntimeError("Image is strange, cannot find right box")
    mid = (start+end) // 2
    res = f(mid)
    if res == 0: return mid
    return binary_search_img(mid+1, end, f, steps_to_stop-1) if res < 0 else \
        binary_search_img(start, mid-1, f, steps_to_stop-1)
# -
# ## Seeding, data preparation


# +
# The images are actually quite big. We will resize to a much smaller size.

bs = 64  # smaller batch size is better for training, but may take longer
sz = 224  # transformed to this size
# -
# +
# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
    os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")

os.listdir('../input')
# -

# +
import pdb
#import torchsnooper
#import pysnooper


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_for_party():
    print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
    if not random_world:
        SEED = 999
        seed_everything(SEED)

prepare_for_party()

def check_image_list_dist(subfolder):
    path_data = Path('../input/aptos2019-blindness-detection')

    il = ImageList.from_folder(path_data/subfolder)

    img_size_stats = {}
    target_blk = []
    for img in il:
        s = tuple(img.size)
        stat = img_size_stats.setdefault(s, 0)
        img_size_stats[s] = stat + 1
        if s[0] == 480:
            bc=get_diag_black_cnt(img)
            target_blk.append(bc)
            print(bc)
    return img_size_stats, target_blk

test_img_size_stats = {(1958, 2588): 134,
                       (480, 640): 1403,  # 3/4
                       (1736, 2416): 225,
                       (1050, 1050): 69,
                       (1944, 2896): 11,
                       (1110, 1467): 2,
                       (1944, 2592): 6,
                       (614, 819): 45,  # around 3/4
                       (576, 768): 2,
                       (1536, 2048): 28,
                       (1117, 1476): 2,
                       (1764, 2146): 1}
# check_image_list_dist('train_images')
train_img_size_stats = {(2136, 3216): 410,
                        (1958, 2588): 533,
                        (1536, 2048): 351,
                        (2848, 4288): 52,
                        (1736, 2416): 638,
                        (1050, 1050): 974,
                        (1000, 1504): 92,
                        (614, 819): 287,
                        (1226, 1844): 61,
                        (1424, 2144): 28,
                        (1944, 2896): 34,
                        (480, 640): 42,
                        (2588, 3388): 141,
                        (1117, 1476): 14,
                        (1110, 1467): 2,
                        (358, 474): 2,
                        (1764, 2146): 1}
# way too different !!!!
# so we just take care of the types with > 30 imgs
# for test 480*640 images, the get_diag_black_cnt number is (137.77405559515324, 3.369116330637486)

# to preprocess our data: we do this:
# 1. cv2 find the contour
# 2. find the center
# 3. use 3/4 crop, binarysearch, until the black margin we wanted is got (we can count the black pixels around the diagonal)
# 4. done
def get_diag_black_cnt(img):
    assert img.data.dtype == torch.float32
    img_r = img.data[0]  # only need to look R channle, most information

    thresh = img_r.mean()/10  # in tensor the value is in range 0 ~ 1
    def blk_cnt(t):
        return sum((dig<thresh).sum() for dig in [t.diagonal(),t.diagonal(offset=5),t.diagonal(offset=-5)])
    img_r_flip = img_r.flip((1))

    return blk_cnt(img_r) + blk_cnt(img_r_flip)


def get_tensor_diag_black_cnt(img):
    assert img.dtype == torch.float32

    img_r = img[0]  # only need to look R channle, most information

    thresh = img_r.mean()/10  # in tensor the value is in range 0 ~ 1
    def blk_cnt(t):
        return sum((dig<thresh).sum() for dig in [t.diagonal(),t.diagonal(offset=5),t.diagonal(offset=-5)])
    img_r_flip = img_r.flip((1))

    return blk_cnt(img_r) + blk_cnt(img_r_flip)


def fastai_img_2_cv2(img): return (image2np(img.data)*255).astype(np.uint8)


def find_center(image, cnt_thresh=20):  # return list, need to post process, should be around center
    """ Thanks to https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/ """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, cnt_thresh, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    cs = []
    clen = np.array([len(c) for c in cnts])
    mid = np.argmax(clen)
    
    for c in [cnts[mid]]:
        # compute the center of the contour
        if len(c) < 5:
            return None
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cs.append((cX, cY, c))
    return cs


def find_DR_center(img, thresh=20):
    cs = find_center(fastai_img_2_cv2(img), cnt_thresh=thresh)
    if cs is None:
        return None, None, None
    h, w = img.size
    for cx, cy, c in cs:
        if w/3 <= cx <= 2*w/3 and h/3 <= cy <= 2*h/3:
            return cx, cy, c  # width, height correspondingly
    return None, None, None


def find_box_from_center(img, ratio=3/4, bc=137.774, bc_std=3.369, thresh=20):  # should put in a class
    """ratio here, 3 height - 4 width (3y, 4x)
    """
    h, w = img.size

    def contour_range(cx, cy, c):
        xs = c[:,0,0]
        ys = c[:,0,1]
        xs_rel = xs - cx
        ys_rel = ys - cy
        ratio = xs_rel*3 - ys_rel*4  # minimum one
        sid = np.argsort(np.absolute(ratio))
        xid = sid[0]
        xid_d = None
        for i in sid[1:]:
            if abs(i-xid) < 5:  # just use the trick, as they can't be that close
                continue
            else:
                xid_d = i
                break
        if xid_d is None:
            raise RuntimeError("Image Strange, cannot find right contour points")
        c1, c2 = (xs[xid], ys[xid]), (xs[xid_d], ys[xid_d])
        if c1[0]*c1[1] > c2[0]*c2[0]:
            c2, c1 = c1, c2
        return xs.max(), xs.min(), ys.max(), ys.min(), c1, c2

    cx, cy, c = find_DR_center(img, thresh=thresh)
    if cx is None: return None
    xmax, xmin, ymax, ymin, cross, cross2 = contour_range(cx, cy, c)

    def find_box_from_cross_points(c1,c2, ratio, expand=28/400):
        # for 480*640, 137.7 ... we just test more times...
        #138/6 = 23, and the r is around 480/2/3*5 = 800/2=400, so it is about 25/400 more
        # so for h,w, both add these
        width_inner = c2[0] - c1[0]
        width_e = width_inner * expand/2
        hight_e = width_e*ratio
        width_e = int(width_e)
        hight_e = int(hight_e)
        c1[0] -= width_e
        c2[0] += width_e
        c1[1] -= hight_e
        c2[1] += hight_e
        return c1, c2
    def clip(h,w, c):
        x=c[0]
        y=c[1]
        x = x if x > 0 else 0
        y = y if y > 0 else 0
        x = x if x < w else w
        y = y if y < h else h
        return x, y
    c1,c2 = find_box_from_cross_points(list(cross), list(cross2), ratio)
    c1 = clip(h,w,c1)
    c2 = clip(h,w,c2)

    bbox = [c1[1], c1[0],c2[1], c2[0]]

    cnt = get_tensor_diag_black_cnt(img.data[:,c1[1]:c2[1],c1[0]:c2[0]])
    print(cnt)
    if cnt == 0:
        print("Image Strange, cannot find black points in diagonals")

    return ImageBBox.create(h,w, [bbox], labels=[0], classes=['ROI']), bbox
    #ch = cy
    #cw = cx
    #h_range = min(ch-ymin, ymax-ch)  # half box height up limit
    #h_start = abs(ch-cross[1])
    #if h_start > h_range:
    #    raise RuntimeError("Image is strange, cannot find right box")
    #w_range = min(cw-xmin, xmax-cw)
    # start from a more likely place, 2/h_range

    #def check_blk_cnt_center_crop(h_add):
    #    w_add = int(h_add/ratio)
    #    cnt = get_tensor_diag_black_cnt(img.data[:,ch-h_add:ch+h_add,cw-w_add:cw+w_add])
    #    ub = (bc+bc_std)/480
    #    lb = (bc-bc_std)/480
    #    r = cnt.float() / (2*h_add)
    #    if r > ub: return 1
    #    if r < lb: return -1
    #    return 0

    # max_rounds = math.floor(math.log2(h_range-h_start))
    # h_proposed = binary_search_img(h_start, h_range, check_blk_cnt_center_crop, max_rounds)
    # binary search in this situation is not stable, as some image width not enough, so the
    # diagonal calculation will be incorrect
    #
    # easier way, just use the cross points, and add a few pixel margin
    #w_ = int(h_proposed * 4/3)
    #w_ = w_ if w_ < w_range else w_range
    # top left bottem rightb    
    #return ImageBBox.create(h,w, [[ch-h_proposed, cw-w_, ch+h_proposed, cw+w_]], labels=[0], classes=['ROI']), [ch-h_proposed, cw-w_, ch+h_proposed, cw+w_]
    #return [ch-h_proposed, cw-w_, ch+h_proposed, cw+w_]

# -
# +
errorsones=[]  # to put together and need to handle later
def check_crop(subfolder):
    path_data = Path('../input/aptos2019-blindness-detection')
    cwd = Path('.')
    topil = ToPILImage()

    il = ImageList.from_folder(path_data/subfolder)
    to_save_path = cwd/(subfolder+'_cropped')
    to_save_path_resized = cwd/(subfolder+'_cropped'+'_resized')
    to_save_path_resized.mkdir()
    to_save_path.mkdir()

    cotinue_f=False
    for i, img in enumerate(il):
        fname = il.items[i]
        name_before_resize = to_save_path/fname.name
        name_after_resize = to_save_path_resized/fname.name

        if fname.name != '81914ceb4e74.png' and cotinue_f: 
            continue
        if cotinue_f:
            cotinue_f = False
            continue

        s = tuple(img.size)
        if s[0] != 480:
            try:
                _, coords = find_box_from_center(img)
                print(coords, name_before_resize)
                pilimg=topil(img.data[:,coords[0]:coords[2],coords[1]:coords[3]])
                pilimg.save(name_before_resize)
                pilimg_r = pilimg.resize((640, 480))  # weight, height
                pilimg_r.save(name_after_resize)
                del pilimg_r
                del pilimg
            except Exception as e:
                print(e)
                print(fname.name, " error happened")
                errorsones.append(fname.name)
            #break
        else:
            copyfile(fname, name_before_resize)
            copyfile(fname, name_after_resize)
from tqdm import tqdm
errorsones=[]  # to put together and need to handle later
def check_crop_namelist(ns):
    subfolder='train_images'
    path_data = Path('../input/aptos2019-blindness-detection')/subfolder
    cwd = Path('.')
    topil = ToPILImage()

    to_save_path = cwd/(subfolder+'_cropped')
    to_save_path_resized = cwd/(subfolder+'_cropped'+'_resized')
    to_save_path_resized.mkdir(parents=True, exist_ok=True)
    to_save_path.mkdir(parents=True, exist_ok=True)

    for imgname in tqdm(ns):
        fname = path_data/imgname
        name_before_resize = to_save_path/fname.name
        name_after_resize = to_save_path_resized/fname.name

        img = open_image(fname)
        s = tuple(img.size)
        if s[0] != 480:
            try:
                _, coords = find_box_from_center(img, thresh=10)  # use small thresh, maybe helpful -> much better
                print(coords, name_before_resize)
                pilimg=topil(img.data[:,coords[0]:coords[2],coords[1]:coords[3]])
                pilimg.save(name_before_resize)
                pilimg_r = pilimg.resize((640, 480))  # weight, height
                pilimg_r.save(name_after_resize)
                del pilimg_r
                del pilimg
            except Exception as e:
                print(e)
                print(fname.name, " error happened")
                errorsones.append(fname20.name)
            #break
        else:
            copyfile(fname, name_before_resize)
            copyfile(fname, name_after_resize)

still_need_to_handle = [
 '033f2b43de6d.png',
 '3a6e9730b298.png',
 'df4913ca3712.png',
 'f002ce614c59.png',
 '4860f7813654.png',
 'ac720570dd0f.png',
 '807135cbc438.png',
 'e1418d28d668.png',
 '42a67337fa8e.png',
 '17d997fe1090.png',
 '64fedbf97473.png',
 'd51b3fe0fa1b.png',
 '6f4719c6bb4b.png',
 'c58971bcebb2.png',
 '5548a7961a3e.png',
 '7214fc7cbe03.png',
 '7269a1d84a57.png',
 '2c77bf969079.png',
 'e821c1b6417a.png',
 'b665041e1633.png',
 '417f408ee8e0.png',
 '6b7cf869622a.png',
 '66d2ca47aa44.png',
 '6cb96a6fb029.png',
 '50d8a8fb7737.png',
 '541db13517e2.png',
 '5b301a6d1ac7.png',
 'f64214bed40e.png',
 '2bb3c492d6d3.png']
#check_crop_namelist(still_need_to_handle)
# -


def prepare_train_dev_df(df, cls_overlap="None_for_fine"):
    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir, '{}.png'.format(x)))
    df = df.drop(columns=['id_code'])
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
    #['normal', 'NPDR_1', 'NPDR_2', 'NPDR_3', 'PDR']
    DR_fine_one_hot = pd.get_dummies(df['diagnosis'], prefix='DR')
    df = pd.concat([df, DR_fine_one_hot], axis=1)

    df['val'] = False
    for i in range(5):
        name = f'DR_{i}'
        val_sub_part = (df[name][df[name]==1]).sample(frac=0.2, replace=False)
        df.loc[val_sub_part.index, 'val'] = True

    if cls_overlap == 'None_for_fine':  # to preserve more information
        pass
    else:
        df['DR_3'][df['diagnosis'] == 4] = 1  # could use label smooth, or mixup
        df['DR_1'][df['DR_2'] == 1] = 1
        df['DR_1'][df['DR_3'] == 1] = 1
        df['DR_2'][df['DR_3'] == 1] = 1

    NPDR_index = (df['diagnosis'] < 4) & (df['diagnosis'] > 0)

    df['diagnosis_coarse'] = 1
    df['diagnosis_coarse'][df['diagnosis'] == 0] = 0  # Normal
    df['diagnosis_coarse'][NPDR_index] = 1            # NPDR
    df['diagnosis_coarse'][df['diagnosis'] == 4] = 2  # PDR

    DR_coarse_one_hot = pd.get_dummies(df['diagnosis_coarse'], prefix='DRC')
    #DR_coarse_one_hot['DRC_1'][DR_coarse_one_hot['DRC_2'] == 1] = 1

    # score_index = NPDR_index | (df['diagnosis']==4)
    # df['NPDR_score'] = np.NaN
    # df['NPDR_score'][score_index] = df['diagnosis'][score_index]
    # todo use whole data, i.e., normal + PDR

    df['NPDR_score'] = df['diagnosis']

    df = pd.concat([df, DR_coarse_one_hot], axis=1)

    len_df = len(df)
    print(f"There are {len_df} images")

    return df


# +
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir, 'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
if use_less_train_data:
    df = df.sample(frac=0.1).copy()

df = prepare_train_dev_df(df)
#newpath = "../input/dr-cropped/dr_train_images_cropped_resized/train_images_cropped_resized/"


def replace_image_cropped(df, from_folder, to_folder):
    # path, f62b8a076833.png is truncted, so remove it
    df['path'] = df['path'].str.replace(from_folder, to_folder, regex=False)
    return df[~df['path'].str.contains('f62b8a076833')]



df = replace_image_cropped(df,
    'aptos2019-blindness-detection/train_images',
    #'dr-cropped/dr_train_images_cropped_resized/train_images_cropped_resized')
    'dr-cropped/dr_train_images_cropped')


# This is actually very small. The [previous competition](https://kaggle.com/c/diabetic-retinopathy-detection) had ~35k images, which supports the idea that pretraining on that dataset may be quite beneficial.

# The dataset is highly imbalanced, with many samples for level 0, and very little for the rest of the levels.

df['diagnosis'].hist(figsize=(10, 5))  # so might be overfitting!!!

# Let's look at an example image:
from PIL import Image

im = Image.open(df['path'][0])
width, height = im.size
print("1st image size: ", width, height)
#im
# -

# +
class CategoryWithScoreProcessor(MultiCategoryProcessor):
    "`PreProcessor` that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds: ItemList, one_hot: bool=False):
        super(CategoryWithScoreProcessor, self).__init__(ds, one_hot)

    def process_one(self, item):
        if self.one_hot or isinstance(item, EmptyLabel): return item
        return item.astype(np.int)

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        raise RuntimeError('For DR, we handle differently, so we don\'t use this function.')


class DRCategory(Category):
    """
    DRCategory just save the raw data as index
    """
    def __int__(self):  return int(self.data[-1])


class DRCategoryListWithScore(MultiCategoryList):
    """
    Data format: e.g. [0,1,0,  1,0,0, 1], first 3 is for coarse classification, second 3 for fine
    classification, the last for regresstion
    """
    _processor = CategoryWithScoreProcessor

    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        super().__init__(items, classes=classes, label_delim=label_delim,one_hot=one_hot, **kwargs)
        #self.loss_func = BCEWithLogitsFlat()
        #self.one_hot = one_hot
        #self.copy_new += ['one_hot']
        self.loss_func = None
        self.thresholds = [0.5, 1.5, 2.5, 3.5]    

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return DRCategory(o, self.classes[o[-1]])

    def analyze_pred(self, pred, thresh: float = 0.):
        # need to analyze which method got better result
        p = (pred >= thresh).float()
        p[-1] = pred[-1]
        return p
        #p = pred.clone().detach()
        #p[:6] = 0

        #coarse_predict = pred[:3].argmax()
        #p[coarse_predict] = 1.
        #if coarse_predict == 1:  # NPDR
        #    fine_predict = pred[3:6].argmax()
        #    #type_pred = fine_predict + 1  
        #    # todo use reg value to analyze. but need to know the threshold
        #    p[3+fine_predict] = 1.
        #return p
        #else:
            #type_pred = coarse_predict if coarse_predict == 0 else 4
        #    return p[]

    def reconstruct(self, t):
        # in ItemList you can see this
        # def analyze_pred(self, pred:Tensor):
        #     "Called on `pred` before `reconstruct` for additional preprocessing."
        reg = t[-1]
        type_pred = 0
        for thr in self.thresholds:
            if reg > thr:
                type_pred += 1  # todo, check if classification info are useful or not
            else:
                break

        return DRCategory(t, self.classes[type_pred])
        # coarse_predict = t[:3].argmax()
        # if coarse_predict == 1:  # NPDR
        #     fine_predict = t[3:6].argmax()
        #     type_pred = fine_predict + 1  
        # else:
        #     type_pred = coarse_predict if coarse_predict == 0 else 4

        # return DRCategory(t, self.classes[type_pred])
# -
# +

src = (ImageList.from_df(df=df, path='./', cols='path')  # get dataset from dataset
        .split_by_idx(df[df['val']==1].index.tolist())  # Splitting the dataset
        .label_from_df(cols=['DRC_0', 'DRC_1', 'DRC_2', 'DR_1', 'DR_2', 'DR_3', 'NPDR_score'], label_cls=partial(DRCategoryListWithScore, classes=['normal', 'NPDR_1', 'NPDR_2', 'NPDR_3', 'PDR'], one_hot=False))  # obtain labels from the level column
       ) # LabelList = ImageList + LabelList
twenty_per_size = int(df['val'].sum())
val_bs = twenty_per_size if twenty_per_size < 800 else 512

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180, max_warp=None,
                      max_zoom=0.9, p_affine=0.5, max_lighting=0.1,
                      p_lighting=0.5)
data = (src.transform(tfms, size=sz, resize_method=ResizeMethod.CROP, padding_mode='reflection')  # Data augmentation
        .databunch(bs=bs, val_bs=val_bs, num_workers=2)  # DataBunch
        )

def get_train_stats(databunch):
    """
    get mean and std of the training set
    """
    tdl = databunch.train_dl

    stats = None
    n = 0
    for x, _ in tdl:
        if stats is None:
            stats = x.new_zeros((2, 3))

        b_n = x.shape[0]
        n += b_n
        # need to record E(X^2) and E(X), for x^2
        stats[0] += x.mean(dim=(0, 2, 3))*b_n
        stats[1] += x.std (dim=(0, 2, 3))*b_n
        print(stats)
    stats /= n
    return stats

#stats = ([0.4285, 0.2286, 0.0753], [0.2700, 0.1485, 0.0812])  # before crop
stats = ([0.5744, 0.3018, 0.0930], [0.1676, 0.1002, 0.0805])  # after crop

if stats is None:
    stats = get_train_stats(data)
    torch.save(stats, 'data_stats.pkl')

DR_img_stats = stats
#imagenet_stats is very different
data = data.normalize((DR_img_stats[0], DR_img_stats[1]))  # Normalize just to mean, std of the data, as it is way too different with imagenet
# this operation will add a transform to data pipeline
# -

data.show_batch(rows=3, figsize=(7,6))

# +
train_dev_ratio = len(data.dl(DatasetType.Train).x) / len(data.dl(DatasetType.Valid).x)
assert 3.9 < train_dev_ratio < 4.1  # make sure it splits correctly
# -


# ## Training (Transfer learning)

# The Kaggle competition used the Cohen's quadratically weighted kappa so I have that here to compare. This is a better metric when dealing with imbalanced datasets like this one, and for measuring inter-rater agreement for categorical classification (the raters being the human-labeled dataset and the neural network predictions). Here is an implementation based on the scikit-learn's implementation, but converted to a pytorch tensor, as that is what fastai uses.

# ### metric

# #### Optimize the Metric

# Optimizing the quadratic kappa metric was an important part of the top solutions in the previous competition. Thankfully, @abhishek has already provided code to do this for us. We will use this to improve the score.


# +
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y, cls_weight):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        sample_weight = y.new_zeros(y.size(), dtype=torch.float)
        reg_predict = y.clone().detach()
        inds = [torch.nonzero(reg_predict == type_id).squeeze(1) for type_id in range(5)]
        for type_id, ind in enumerate(inds):
            sample_weight[ind] = cls_weight[type_id]

        ll = cohen_kappa_score(y, X_p, weights='quadratic', sample_weight=sample_weight)
        return -ll

    def fit(self, X, y, cls_weight):
        loss_partial = partial(self._kappa_loss, X=X, y=y, cls_weight=cls_weight)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif coef[0] <= pred < coef[1]:
                X_p[i] = 1
            elif coef[1] <= pred < coef[2]:
                X_p[i] = 2
            elif coef[2] <= pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

# -

# +
def convert_to_normal_pred(pred, thresholds):  # for batched data, still is the batch size, (64,7)
    thresh_for_PDR = 3.

    if len(pred.shape) == 1:
        #if pred[2] > 0 and pred[-1] > thresh_for_PDR:
        #    return 4
        #coarse_predict = pred[:3].argmax()
        #if coarse_predict == 1:  # NPDR, logit wrong..., argmax need for softmax... and this will ignore all the PDR thing
        #    fine_predict = pred[3:6].argmax()
        #    return fine_predict + 1  # todo use reg value to analyze. but need to know the threshold
        #else:
        #    return coarse_predict if coarse_predict == 0 else 4

        reg = pred[-1]
        type_pred = 0
        for thr in thresholds:
            if reg > thr:
                type_pred += 1  # todo, check if classification info are useful or not
            else:
                break
        return type_pred
    else:
        #coarse_predict = pred[:,:3].argmax(dim=1)  # just do this, and cross our finger for the PDR v.s. NPDR will predict properly.

        #predict = coarse_predict.clone().detach()

        #NPDR_inds_subset = torch.nonzero(coarse_predict == 1).squeeze(1)  # for multi label... how do we do?
        #if len(NPDR_inds_subset) > 0:
        #    fine_predict = pred[NPDR_inds_subset,3:6].argmax(dim=1)
        #    fine_predict += 1  # todo use reg value to analyze. but need to know the threshold

        #    predict[NPDR_inds_subset] = fine_predict

        #reg_value = pred[:,-1]
        #PDR_logit = pred[:,2]

        #PDR_subset = torch.nonzero((PDR_logit > 0) * (reg_value > thresh_for_PDR)).squeeze(1)
        #predict[PDR_subset] = 4
        reg_predict = pred[:, -1].clone().detach()
        inds = [torch.nonzero(reg_predict > thr).squeeze(1) for thr in thresholds]
        predict = reg_predict.new_zeros(reg_predict.size(), dtype=torch.int64)
        for type, ind in enumerate(inds):
            predict[ind] = type+1

        del reg_predict

        return predict


def quadratic_kappa(y_hat, y, cls_weight=None, y_hat_predicted=False):
    # need to convert our answer format

    if not y_hat_predicted:
        coefficients = [0.5, 1.5, 2.5, 3.5]
        y_hat = convert_to_normal_pred(y_hat, coefficients)

    if len(y.shape) < 2:
        target = y
    else:
        target = y[:, -1].int()

    y = target
    sample_weight = y.new_ones(y.size(), dtype=torch.float)

    if cls_weight is not None:
        reg_predict = y.clone().detach()
        inds = [torch.nonzero(reg_predict == type_id).squeeze(1) for type_id in range(5)]
        for type_id, ind in enumerate(inds):
            sample_weight[ind] = cls_weight[type_id]

    return torch.tensor(cohen_kappa_score(y_hat, target, weights='quadratic', sample_weight=sample_weight), device='cuda:0')
# -


# ### loss


# +
# modified based https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class DR_FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0., eps=1e-7, from_logits=True,
                 cls_cnt=None, cls_overlap="None_for_fine"):
        super(DR_FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

        self.from_logits = from_logits

        if cls_cnt is not None:
            normal_cnt = cls_cnt[0]

            NPDR_1_cnt = cls_cnt[1]
            NPDR_2_cnt = cls_cnt[2]
            NPDR_3_cnt = cls_cnt[3]

            PDR_cnt = cls_cnt[4]
        else:
            normal_cnt = 1805

            NPDR_1_cnt = 370
            NPDR_2_cnt = 999
            NPDR_3_cnt = 193

            PDR_cnt = 295

        s = normal_cnt + NPDR_1_cnt + NPDR_2_cnt + NPDR_3_cnt + PDR_cnt
        fine_s = NPDR_1_cnt + NPDR_2_cnt + NPDR_3_cnt

        def cal_neg_coef(a,b,c, s=None):  # s can be passed if there is overlay in abc
            if s is None: s = a + b + c
            return [a/(s-a), b/(s-b), c/(s-c)]

        def cal_ratio(a,b,c,d):  # s can be passed if there is overlay in abc
            return [a/b, a/c, a/d]

        #b_n_r, b_npdr_r, b_pdr_r = cal_ratio(normal_cnt, normal_cnt, fine_s, PDR_cnt)
        #b_npdr1_r_not_added, b_npdr2_r_not_added, b_npdr3_r_not_added = \
        #    cal_ratio(normal_cnt, NPDR_1_cnt, NPDR_2_cnt, NPDR_3_cnt)
        #self.coarse_mag = [b_n_r, b_npdr_r, b_pdr_r]  # only [-1] is used
        #self.fine_mag_not_added = [b_npdr1_r_not_added, b_npdr2_r_not_added, b_npdr3_r_not_added]
        base_cnt = NPDR_2_cnt

        self.coarse_mag = cal_ratio(base_cnt, normal_cnt, fine_s, PDR_cnt)
        self.fine_mag_not_added = \
            cal_ratio(base_cnt, NPDR_1_cnt, NPDR_2_cnt, NPDR_3_cnt)
        self.reg_mag = 2.  # overall loss for regression

        # following for alpha balancer
        if cls_overlap == "None_for_fine":
            NPDR_3_added = NPDR_3_cnt
            NPDR_2_added = NPDR_2_cnt
            NPDR_1_added = NPDR_1_cnt
        else:
            NPDR_3_added = NPDR_3_cnt + PDR_cnt  # PDR as NPDR3 ... not very good, anyway
            NPDR_2_added = NPDR_2_cnt + NPDR_3_added
            NPDR_1_added = NPDR_2_cnt + NPDR_3_added + NPDR_1_cnt

        NPDR_cnt_added_PDR = fine_s  # + PDR_cnt
        #a_normal, a_NPDR, a_PDR = \
        self.coarse_a = \
            cal_neg_coef(normal_cnt, NPDR_cnt_added_PDR, PDR_cnt, s)
        self.fine_a = \
            cal_neg_coef(NPDR_1_added, NPDR_2_added, NPDR_3_added, NPDR_cnt_added_PDR)
        # only used in balancing classification
        self.fine_mag = \
            cal_ratio(base_cnt, NPDR_1_added, NPDR_2_added, NPDR_3_added)

        print('mag coef: ',
              [self.coarse_mag[0]] + self.fine_mag_not_added + [self.coarse_mag[-1]],
              "\nalpha balancer here will be multiplied by negtive loss part\n",
              self.coarse_a, self.fine_a, self.fine_mag)

    @staticmethod
    def cal_balance_ratio_2_parts(pos, neg):
        pos_ratio = neg/(pos+neg)
        return pos_ratio, 1-pos_ratio, pos*pos_ratio

    def forward(self, input, target, **kwargs):
        # so we have 6 logits and one tensor for regression
        reduction_param = kwargs.get('reduction', 'mean')
        target = target.float()  # otherwise, int*float thing

        reg_score_pred = input[:, -1]
        if self.from_logits:
            cls_input = torch.sigmoid(input[:, :-1])
        else:
            cls_input = input[:, :-1]

        coarse_cls_input = cls_input[:, :3]
        coarse_target = target[:, :3]

        # todo add loss/constraint for NPDR,PDR classfication? might be helpful, after we analyze our errors!!
        coarse_cls_loss = self.focal_binary_loss(coarse_cls_input, coarse_target, self.gamma, self.coarse_a, self.eps,
                                      self.coarse_mag, reduction=reduction_param)
        if reduction_param == 'none':  # need to squash for coarse and fine classes
            coarse_cls_loss = coarse_cls_loss.sum(dim=1)

        loss = coarse_cls_loss

        NPDR_inds_subset = torch.nonzero(coarse_target[:, 1] > 0).squeeze(1)
        fine_cls_input = cls_input[NPDR_inds_subset, 3:6]
        fine_target = target[NPDR_inds_subset, 3:6]
        fine_cls_loss = self.focal_binary_loss(fine_cls_input, fine_target, self.gamma, self.fine_a, self.eps, self.fine_mag,
                                       reduction=reduction_param)
        if reduction_param == 'none':  # need to squash for coarse and fine classes
            fine_cls_loss = fine_cls_loss.sum(dim=1)
            loss[NPDR_inds_subset] += fine_cls_loss  # size ...
        else:
            loss += fine_cls_loss  # a value

        # for reg loss, we do class balance too
        reg_loss = self.reg_mag * F.smooth_l1_loss(reg_score_pred, target[:, -1], reduction='none')

        NPDR1_inds_subset = torch.nonzero(target[:, -1] == 1).squeeze(1)
        NPDR2_inds_subset = torch.nonzero(target[:, -1] == 2).squeeze(1)
        NPDR3_inds_subset = torch.nonzero(target[:, -1] == 3).squeeze(1)
        reg_loss[NPDR1_inds_subset] *= self.fine_mag_not_added[0]
        reg_loss[NPDR2_inds_subset] *= self.fine_mag_not_added[1]
        reg_loss[NPDR3_inds_subset] *= self.fine_mag_not_added[2]

        #normal_inds_subset = torch.nonzero(coarse_target[:, 0] > 0).squeeze(1)
        #reg_loss[normal_inds_subset] *= a_normal

        PDR_inds_subset = torch.nonzero(coarse_target[:, 2] > 0).squeeze(1)
        reg_loss[PDR_inds_subset] *= self.coarse_mag[-1]

        if reduction_param != 'none':
            reg_loss = torch.mean(reg_loss) if reduction_param == 'mean' else torch.sum(reg_loss)
        loss += reg_loss

        return loss

    @staticmethod
    def focal_binary_loss(input, target, gamma, alpha, eps, mag, reduction='mean'):
        y_pred = input
        y = target
        not_y = 1 - y
        not_y_pred = 1 - y_pred

        y_pred = y_pred.clamp(eps, 1. - eps)
        not_y_pred = not_y_pred.clamp(eps, 1. - eps)

        if eps > gamma > -eps:  # 0
            pos_gamma_balancer = 1.
            neg_gamma_balancer = 1.
        elif eps > gamma - 1 > -eps:
            pos_gamma_balancer = not_y_pred
            neg_gamma_balancer = y_pred
        else:
            pos_gamma_balancer = not_y_pred ** gamma
            neg_gamma_balancer = y_pred ** gamma

        # RuntimeError: expected backend CUDA and dtype Float but got backend CUDA and dtype Long
        loss = -                       pos_gamma_balancer * y * torch.log(y_pred)  # cross entropy
        loss += -y.new_tensor(alpha) * neg_gamma_balancer * not_y * torch.log(not_y_pred)
        loss *= loss.new_tensor(mag)

        #if target.requires_grad:  # we don't care about this, so just ignore
        
        if reduction != 'none':
            return torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        #else:
        #    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        #    ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
        #return ret
        return loss
# -


# ### learner (resnet 50)


# +
# **Training:**
#
# We use transfer learning, where we retrain the last layers of a pretrained neural network. I use the ResNet50 architecture trained on the ImageNet dataset, which has been commonly used for pre-training applications in computer vision. Fastai makes it quite simple to create a model and train:
def create_DR_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                   concat_pool:bool=True, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def DR_learner(data: DataBunch, base_arch: Callable, cut: Union[int, Callable] = None, pretrained: bool = True,
               lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5, custom_head: Optional[nn.Module] = None,
               split_on: Optional[SplitFuncOrIdxList] = None, bn_final: bool = False, init=nn.init.kaiming_normal_,
               concat_pool: bool = True, attention=False, **kwargs: Any) -> Learner:
    "Build convnet style learner."
    meta = cnn_config(base_arch)

    "Create custom convnet architecture"
    body = create_body(base_arch, pretrained, cut)
    if custom_head is None:  # quadnet head
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        if attention:
            pass
        head = create_DR_head(nf, data.c+2, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    else:
        head = custom_head

    model = nn.Sequential(body, head)

    learn = Learner(data, model, **kwargs)
    learn.split(split_on or meta['split'])
    if pretrained: learn.freeze()
    if init: apply_init(model[1], init)
    return learn
# -


# +
cls_cnt = df['diagnosis'].value_counts().sort_index().values
fl_normal = DR_FocalLoss(gamma=0., cls_cnt=cls_cnt)  # changed lr decay 2/0.15 + patience=3, do not use focal loss...
# set_trace() we convert back to ipynb
# learner = DR_learner(data, vision.models.densenet121, metrics=[quadratic_kappa])
#learn = cnn_learner(data, base_arch=models.resnet50, metrics=[quadratic_kappa])


def get_cls_weight(cls_cnt, multiply=[1, 1, 1, 1, 1]):
    cls_weight = cls_cnt[0]/cls_cnt
    # cls_weight = [1,1,1,1,1]
    assert len(cls_cnt) == len(multiply)
    for i, m in enumerate(multiply):
        cls_weight[i] *= m

    return cls_weight


#cls_weight = get_cls_weight(cls_cnt, [0.5, 1.5, 3, 1, 0.5])
cls_weight = get_cls_weight(cls_cnt)

learn = DR_learner(data, vision.models.resnet50, cut=-1, loss_func=fl_normal,
                   metrics=[partial(quadratic_kappa, cls_weight=cls_weight)],
                   callback_fns=[partial(CSVLogger, append=True)])

# -


# ### train


# +
def train_triangular_lr(learn, inner_step=0, cycle_cnt=None, max_lr=None, loss=None):
    if inner_step == 0:
        if loss is not None:
            learn.loss_func = loss
        learn.freeze()
        learn.lr_find()
        learn.recorder.plot(suggestion=True)
    elif inner_step == 1:
        assert max_lr is not None and cycle_cnt is not None
        learn.freeze()
        learn.fit_one_cycle(cycle_cnt, max_lr=max_lr, wd=0.1)
        learn.recorder.plot_losses()
        learn.recorder.plot_metrics()
        learn.save('dr-stage1')
    elif inner_step == 2:
        learn.unfreeze()
        learn.lr_find()
        learn.recorder.plot(suggestion=True)
    elif inner_step == 3:
        assert max_lr is not None and cycle_cnt is not None
        # Min numerical gradient: 1.91E-06
        #learn.fit_one_cycle(6, max_lr=slice(1e-6, 5e-6/5))
        learn.unfreeze()
        learn.fit_one_cycle(cycle_cnt, max_lr=max_lr, wd=0.1)
    
        learn.recorder.plot_losses()
        learn.recorder.plot_metrics()
        learn.save('dr-stage2')
# -

# +
savedPath = Path('models/stage-2.pth')
if savedPath.exists():
    learn.load('stage-2')
# -


# +
# !echo '#!/bin/sh\n( touch metric.log; tail -f history.csv | while true; do nc -v 23.105.212.181 60020; sleep 1; done ) & ' > log_tel.sh
# !chmod +x log_tel.sh
# #!nc -h || apt install netcat -y

from io import StringIO
from subprocess import Popen, PIPE

#cmdstr = 'python -m unittest unitTest.PSKenelTest.test_pytorch_model_dev'
cmdstr = 'sh ./log_tel.sh'
#Popen(cmdstr.split(), stdout=PIPE)

# -
# +
if submitting_to_LB and do_lr_find:
    train_triangular_lr(learn, inner_step=0)
# -

# +
def get_max_lr(learn):
    try:
        rec = learn.recorder
        lrs = rec._split_list(rec.lrs, 10, 5)
        losses = rec._split_list(rec.losses, 10, 5)
        losses = [x.item() for x in losses]

        mg = (np.gradient(np.array(losses))).argmin()
        print(f"Min numerical gradient: {lrs[mg]:.2E}")
        return lrs[mg]
    except:
        print("Failed to compute the gradients, there might not be enough points.")
        return None
# -

# +
stage_1_cycle = 4 if not use_less_train_data else 1

max_lr_stage_1 = None
if submitting_to_LB and do_lr_find:
    max_lr_stage_1 = get_max_lr(learn)

if max_lr_stage_1 is None or max_lr_stage_1 < 5e-4:
    max_lr_stage_1 = 1e-2
train_triangular_lr(learn, inner_step=1, cycle_cnt=stage_1_cycle, max_lr=max_lr_stage_1)  # choose 3.31E-02 as suggested
# -

# + 
if submitting_to_LB and do_lr_find:
    train_triangular_lr(learn, inner_step=2)  # choose 3.31E-02 as suggested
# -

# + 
stage_2_cycle = 6 if not use_less_train_data else 1

max_lr_stage_2 = None
if submitting_to_LB and do_lr_find:
    max_lr_stage_2 = get_max_lr(learn)

if max_lr_stage_2 is None or max_lr_stage_2 < 1e-6:
    max_lr_stage_2 = 5e-5
last_layer_lr_scale_down = 10.

last_layer_max_lr_stage_2 = max_lr_stage_1 / last_layer_lr_scale_down
train_triangular_lr(learn, inner_step=3, cycle_cnt=stage_2_cycle, max_lr=slice(max_lr_stage_2, last_layer_max_lr_stage_2))
# -

# +
# we need to subclass our our interpretor, as existed one use argmax to predict class

class DRClassificationInterpretation(ClassificationInterpretation):
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid, 
                 cls_converter: Callable=None):
        super(DRClassificationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        assert cls_converter is not None
        self.pred_class = cls_converter(self.preds)
        self.y_true = self.y_true[:, -1]

    @classmethod
    def from_learner(cls, learn: Learner,  ds_type:DatasetType=DatasetType.Valid, cls_converter: Callable=None):
        "Gets preds, y_true, losses to construct base class from a learner"
        preds_res = learn.get_preds(ds_type=ds_type, with_loss=True)
        return cls(learn, *preds_res, cls_converter=cls_converter)
# -
# +
# Let's evaluate our model:

coefficients = [0.5, 1.5, 2.5, 3.5]
interp = DRClassificationInterpretation.from_learner(learn, cls_converter=partial(convert_to_normal_pred, thresholds=coefficients))

# +
losses, idxs = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)
# -


# +
interp.preds[idxs[:20]]
# -

# +
interp.plot_top_losses(k=12, heatmap=False)
print(interp.confusion_matrix())
print(cohen_kappa_score(interp.pred_class, interp.y_true, weights='quadratic'))
# /opt/conda/lib/python3.6/site-packages/fastai/vision/learner.py:147
# /opt/conda/lib/python3.6/site-packages/fastai/vision/learner.py
# #!cat -n /opt/conda/lib/python3.6/site-packages/fastai/data_block.py
# -

# +
interp.plot_confusion_matrix(figsize=(12, 12), dpi=98)
# -

# +
# ## TTA
# 
# Test-time augmentation, or TTA, is a commonly-used technique to provide a boost in your score, and is very simple to implement. Fastai already has TTA implemented, but it is not the best for all purposes, so I am redefining the fastai function and using my custom version.
def _tta_only(learn: Learner, ds_type: DatasetType = DatasetType.Valid, num_pred: int = 10) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    aug_tfms = [o for o in learn.data.train_ds.tfms]
    try:
        pbar = master_bar(range(num_pred))
        for i in pbar:
            ds.tfms = aug_tfms
            yield get_preds(learn.model, dl, pbar=pbar)[0]
    finally:
        ds.tfms = old

Learner.tta_only = _tta_only

def _TTA(learn: Learner, beta: float = 0, ds_type: DatasetType = DatasetType.Valid, num_pred: int = 1,
         with_loss: bool = False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds, y = learn.get_preds(ds_type)
    if beta is None:
        all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))
        avg_preds = torch.stack(all_preds).mean(0)
        return preds, avg_preds, y
    else:
        if beta == 1.:
            final_preds = preds
        else:
            avg_preds = torch.stack(all_preds).mean(0)
            all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))
            final_preds = preds * beta + avg_preds * (1 - beta)
        if with_loss:
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss
        return final_preds, y

Learner.TTA = _TTA
# -


# ## predict submission.csv


# +
valid_preds = (interp.preds, interp.y_true)

optR = OptimizedRounder()
# might overfit ...(but at V15 code, without it, performance is really bad)
optR.fit(valid_preds[0][:, -1], valid_preds[1], cls_weight=cls_weight)

coefficients = optR.coefficients()
coefficients
# -

# +
test_predictions = None
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')


def preds_for_test(learn, sample_df, optR, coefficients):
    learn.data.add_test(
        ImageList.from_df(sample_df, '../input/aptos2019-blindness-detection',
                          folder='test_images', suffix='.png'))

    preds, y = learn.TTA(ds_type=DatasetType.Test, beta=1.)
    test_predictions = optR.predict(preds[:, -1], coefficients)
    return test_predictions
# -
# +

if submitting_to_LB:
    if test_predictions is None: test_predictions = preds_for_test(learn, sample_df, optR, coefficients)
    sample_df.diagnosis = test_predictions.astype(int)
    sample_df.to_csv('submission.csv', index=False)

    sample_df.head()
# -

# +
# use coefficients to predict
interp.pred_class = torch.tensor(optR.predict(interp.preds[:, -1],
                                 coefficients).astype(np.int),
                                 dtype=torch.int64)
# -
# +
print(interp.confusion_matrix())
print(quadratic_kappa(interp.pred_class, interp.y_true, cls_weight=cls_weight, y_hat_predicted=True))
# -

