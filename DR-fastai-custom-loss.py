#!/usr/bin/env python
# coding: utf-8

# +
import numpy as np
import pandas as pd
import os
import scipy as sp
from functools import partial
from sklearn import metrics

import torch
from fastai import *
from fastai.core import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.torch_core import *

from fastai.vision import *
from fastai.vision.learner import create_head, cnn_config, num_features_model

from IPython.core.debugger import set_trace

# -

# +

# !apt install tree -y

# !tree -d ../input/

# # todo: check how accuracy is calculated, or just submit and check ans...

# !nvidia-smi

# %reload_ext autoreload
# %autoreload 2

exit()

# !ls -lh models/

# +
import subprocess
import os
import gc

USER_NAME = 'pengyu'
TFRECORD_FILDATA_FLAG = '.tf_record_saved'
GDRIVE_DOWNLOAD_DEST = '/proc/driver/nvidia'


def run_commans(commands, timeout=30):
    for c in commands.splitlines():
        c = c.strip()
        if c.startswith("#"):
            continue
        stdout, stderr = run_process(c, timeout)
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))
            print("stop at command {}, as it reports error in stderr".format(c))
            break


def run_process(process_str, timeout):
    print("{}:{}$ ".format(USER_NAME, os.getcwd())+process_str)
    MyOut = subprocess.Popen(process_str,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    try:
        MyOut.wait(timeout=timeout)
        return MyOut.communicate()
    except subprocess.TimeoutExpired as e:
        return None, bytes('\'{}\''.format(e), 'utf-8')

def run_process_print(process_str,timeout=30):
    stdout, stderr = run_process(process_str, timeout)
    if stdout:
        print(stdout.decode('utf-8'))
    if stderr:
        print(stderr.decode('utf-8'))

def pip_install_thing():
    to_installs = """pip install --upgrade pip
    #pip install -q tensorflow-gpu==2.0.0-alpha0
    #pip install drive-cli"""
    run_commans(to_installs, timeout=60*10)

#pip_install_thing()

def upload_file_one_at_a_time(file_name, saved_name=None):
    if not saved_name:
        saved_name = file_name.split('/')[-1]
    run_process_print("curl  --header \"File-Name:{1}\" --data-binary @{0} http://23.105.212.181:8001".format(file_name, saved_name))
    #print("You need to goto VPS and change filename")

def download_file_one_at_a_time(file_name, directory=".", overwrite=False):
    if overwrite:
        run_process_print("wget http://23.105.212.181:8000/{0} -O \"{1}/{0}\"".format(file_name, directory))
    else:
        run_process_print("[ -f {1}/{0} ] || wget http://23.105.212.181:8000/{0} -P {1}".format(file_name, directory))

#upload_file_one_at_a_time("env_prepare.py")

def setup_kaggle():
    s = """pip install kaggle 
    mkdir $HOME/.kaggle 
    echo '{"username":"k1gaggle","key":"f51513f40920d492e9d81bc670b25fa9"}' > $HOME/.kaggle/kaggle.json
    chmod 600 $HOME/.kaggle/kaggle.json
    """
    run_commans(s, timeout=60)

def download_kaggle_data():
    if not os.path.isfile('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'):
        s = """kaggle competitions download jigsaw-unintended-bias-in-toxicity-classification
        unzip test.csv.zip
        unzip train.csv.zip
        """
        run_commans(s, timeout=60)


def setup_gdrive():
    download_file_one_at_a_time("gdrive")
    s = """chmod +x ./gdrive
    mkdir $HOME/.gdrive 
    chmod +x ./gdrive
    """
    run_commans(s)
    #download_file_one_at_a_time("token_v2.json", "$HOME/.gdrive")
    str= """{
        "access_token": "ya29.GlsWB6DpEzK1qbegW-7FGy84GUtdR8O57aoq3i73DiFLlwpGxG1hZGwCVLiBIFNCDIw0zgQ6Fs4aBkf1YWbc30_yJMLCtv1E1b20nqMF2gRF3cJU_Ks-xnsaF5WV",
        "token_type": "Bearer",
        "refresh_token": "1/uxgj61NZOFM_LkIZd6QHpGX0Nj8bm9004DK68Ywu0pU",
        "expiry": "2019-05-27T06:11:29.604819094-04:00"
    }"""
    with open("token_v2.json", 'wb') as f:
            f.write(bytes(str, 'utf-8'))
    run_process_print('mv token_v2.json $HOME/.gdrive') # last command cannot know $HOME easily, so python + shell

def mount_gdrive():
    from google.colab import drive
    drive.mount('/content/gdrivedata')
    "4/ZQF_RbIHCF9ub34Y9_pEV71pY1TroSCzkssAot-qRmZ8PDTwwV79NQ4"

    run_process_print(f'touch {TFRECORD_FILDATA_FLAG}')


#setup_gdrive()
#upload_file_one_at_a_time("~/.kaggle/kaggle.json")
#upload_file_one_at_a_time("env_prepare.py")

#upload_file_one_at_a_time("/sync/AI/dog-breed/kaggle-dog-breed/src/solver/server.py")
#download_file_one_at_a_time("kaggle.json")
#download_file_one_at_a_time("server.py")

def list_submisstion():
    run_process_print("kaggle competitions submissions -c jigsaw-unintended-bias-in-toxicity-classification")


def get_mem_hogs():
    '''get memory usage by ipython objects

    :return: sorted ipython objects (name, mem_usage) list
    '''
    import sys
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

def download_lstm_from_gdrive():
    """
    download from gdrive, files are in `lstm_data` folder
    """
    run_commans(
        f"""
        #./gdrive download 1V651fAb8_RxDF--VfHWlUQ8wTzULPPyu --path {GDRIVE_DOWNLOAD_DEST} # train
        #./gdrive download 144glCjAb6rTJXNddslpc-mgQBCGGybTr --path {GDRIVE_DOWNLOAD_DEST} # test
        #./gdrive download 1A3vj6mBUTGYnvd4HfDyI9hBT78IWyABf --path {GDRIVE_DOWNLOAD_DEST} # embedding
        #./gdrive download 1d_2uUzStUhuzErWAcIIk2TuzA1bFyKN7 --path {GDRIVE_DOWNLOAD_DEST} # predicts (no res)
        ./gdrive download 1H6ktwj59KtmiUbXZT4bwT7cIUL77MLsa --path {GDRIVE_DOWNLOAD_DEST} # model
        #./gdrive download   # predicts result (for target)
        #./gdrive download   # identity model
        #mv lstm_data/* . 
        touch """ + TFRECORD_FILDATA_FLAG
        ,
        timeout=60*10
    )

def up():
    run_process_print('rm -rf __pycache__ /proc/driver/nvidia/identity-model/events*')
    download_file_one_at_a_time("data_prepare.py", overwrite=True)
    download_file_one_at_a_time("lstm.py", overwrite=True)
    download_file_one_at_a_time("env_prepare.py", overwrite=True)

def exit00():
    import os
    os._exit(00)  # will make ipykernel restart

quick = True
if os.getcwd().find('lstm') > 0:
    #upload_file_one_at_a_time("data_prepare.py")
    setup_gdrive()
else:
    #do_gc()
    if not os.path.isfile('.env_setup_done'):
        try:
            mount_gdrive()
        except ModuleNotFoundError:
            setup_gdrive()
            #download_lstm_from_gdrive()

        if not quick:
            if not os.path.isdir("../input") and not os.path.isdir('/content/gdrivedata/My Drive/'):
                setup_kaggle()
                download_kaggle_data()
                list_submisstion()
            pip_install_thing()
        #run_process_print('export PATH=$PWD:$PATH') # not helpful, subshell
        run_process_print('touch .env_setup_done')

#up()  # this is needed always


#get_ipython().reset()  # run in ipython
# #!wget http://23.105.212.181:8000/lstm.py -O lstm.py && wget http://23.105.212.181:8000/data_prepare.py -O data_prepare.py && python lstm.py
# https://drive.google.com/file/d/1A3vj6mBUTGYnvd4HfDyI9hBT78IWyABf/view?usp=sharing
# https://drive.google.com/open?id=1V651fAb8_RxDF--VfHWlUQ8wTzULPPyu
# https://drive.google.com/open?id=144glCjAb6rTJXNddslpc-mgQBCGGybTr# -


# +

# !./gdrive download 1aDPlZSOCm66f3XkFZH9xEwqHoyLd5eoC ; mkdir models ; mv *pth models

# !pip install torchsnooper
# -

# +
# # %load fastai-custom-loss.py

# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
    os.makedirs('/tmp/.cache/torch/checkpoints/')
#get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


os.listdir('../input')

import pdb
import torchsnooper
import pysnooper

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def pre_prepare_train_dev_data():
    print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
    SEED = 999
    seed_everything(SEED)


def prepare_train_dev_data():
    base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
    train_dir = os.path.join(base_image_dir, 'train_images/')
    df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir, '{}.png'.format(x)))
    df = df.drop(columns=['id_code'])
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
    #['normal', 'NPDR_1', 'NPDR_2', 'NPDR_3', 'PDR']
    prefix = 'DR'
    DR_fine_one_hot = pd.get_dummies(df['diagnosis'], prefix=prefix)
    df = pd.concat([df, DR_fine_one_hot], axis=1)

    df['val'] = False
    for i in range(5):
        name = f'{prefix}_{i}'
        val_sub_part = (df[name][df[name]==1]).sample(frac=0.2, replace=False)
        df.loc[val_sub_part.index, 'val'] = True

    NPDR_index = (df['diagnosis'] < 4) & (df['diagnosis'] > 0)

    df['diagnosis_coarse'] = 0
    df['diagnosis_coarse'][df['diagnosis'] == 0] = 0
    df['diagnosis_coarse'][df['diagnosis'] == 4] = 2
    df['diagnosis_coarse'][NPDR_index] = 1
    DR_coarse_one_hot = pd.get_dummies(df['diagnosis_coarse'], prefix='DRC')
    DR_coarse_one_hot['DRC_1'][DR_coarse_one_hot['DRC_2'] == 1] = 1

    #score_index = NPDR_index | (df['diagnosis']==4)
    #df['NPDR_score'] = np.NaN
    #df['NPDR_score'][score_index] = df['diagnosis'][score_index]  # todo use whole data, i.e., normal + PDR

    df['NPDR_score'] = df['diagnosis']

    df = pd.concat([df, DR_coarse_one_hot], axis=1)

    len_df = len(df)
    print(f"There are {len_df} images")

    return df


# +
pre_prepare_train_dev_data()
df = prepare_train_dev_data()

# This is actually very small. The [previous competition](https://kaggle.com/c/diabetic-retinopathy-detection) had ~35k images, which supports the idea that pretraining on that dataset may be quite beneficial.

# The dataset is highly imbalanced, with many samples for level 0, and very little for the rest of the levels.


df['diagnosis'].hist(figsize=(10, 5))

# Let's look at an example image:


# +
from PIL import Image

im = Image.open(df['path'][1])
width, height = im.size
print(width, height)
#im

# +
# plt.imshow(np.asarray(im))

# The images are actually quite big. We will resize to a much smaller size.

bs = 64  # smaller batch size is better for training, but may take longer
sz = 224  # transformed to this size


# called in this:
# /opt/conda/lib/python3.6/site-packages/fastai/data_block.py(477)_inner()
#     475             self.valid = fv(*args, from_item_lists=True, **kwargs)
#     476             self.__class__ = LabelLists
# --> 477             self.process()
#     478             return self
#     479         return _inner

# /opt/conda/lib/python3.6/site-packages/fastai/data_block.py(530)process()
#     528     def process(self):
#     529         "Process the inner datasets."
# --> 530         xp,yp = self.get_processors()  # here the processor is of type fastai.data_block.MultiCategoryProcessor
#     531         for ds,n in zip(self.lists, ['train','valid','test']): ds.process(xp, yp, name=n)
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
    def __int__(self):  return int(self.data[-1])


class DRCategoryListWithScore(MultiCategoryList):
    "Basic `ItemList` for single classification labels."
    _processor = CategoryWithScoreProcessor

    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        super().__init__(items, classes=classes, label_delim=label_delim,one_hot=one_hot, **kwargs)
        #self.loss_func = BCEWithLogitsFlat()
        #self.one_hot = one_hot
        #self.copy_new += ['one_hot']
        self.loss_func = None

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return DRCategory(o, self.classes[o[-1]])

    def analyze_pred(self, pred, thresh: float = 0.5):
        coarse_predict = pred[:3].argmax()
        if coarse_predict == 1:  # NPDR
            fine_predict = pred[3:6].argmax()
            return fine_predict + 1  # todo use reg value to analyze. but need to know the threshold
        else:
            return coarse_predict if coarse_predict == 0 else 4

    def reconstruct(self, t):
        return DRCategory(t, self.classes[t[-1]])
    

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=360, max_warp=0, max_zoom=1.1, max_lighting=0.1,
                      p_lighting=0.5)
src = (ImageList.from_df(df=df, path='./', cols='path')  # get dataset from dataset
        .split_by_idx((df['val']==1).tolist())  # Splitting the dataset
        .label_from_df(cols=['DRC_0', 'DRC_1', 'DRC_2', 'DR_1', 'DR_2', 'DR_3', 'NPDR_score'], label_cls=partial(DRCategoryListWithScore, classes=['normal', 'NPDR_1', 'NPDR_2', 'NPDR_3', 'PDR'], one_hot=False))  # obtain labels from the level column
       ) # LabelList = ImageList + LabelList
data = (src.transform(tfms, size=sz, resize_method=ResizeMethod.SQUISH, padding_mode='zeros')  # Data augmentation
        .databunch(bs=bs, num_workers=4)  # DataBunch
        .normalize(imagenet_stats)  # Normalize
        )
# -

src.valid.y


# +
# data.show_batch(rows=3, figsize=(7,6))
# -


# ## Training (Transfer learning)

# The Kaggle competition used the Cohen's quadratically weighted kappa so I have that here to compare. This is a better metric when dealing with imbalanced datasets like this one, and for measuring inter-rater agreement for categorical classification (the raters being the human-labeled dataset and the neural network predictions). Here is an implementation based on the scikit-learn's implementation, but converted to a pytorch tensor, as that is what fastai uses.


# +
from sklearn.metrics import cohen_kappa_score

def convert_to_normal_pred(pred):  # for batched data, still is the batch size, (64,7)
    if len(pred.shape) == 1:
        coarse_predict = pred[:3].argmax()
        if coarse_predict == 1:  # NPDR
            fine_predict = pred[3:6].argmax()
            return fine_predict + 1  # todo use reg value to analyze. but need to know the threshold
        else:
            return coarse_predict if coarse_predict == 0 else 4
    else:
        coarse_predict = pred[:,:3].argmax(dim=1)  # just do this, and cross our finger for the PDR v.s. NPDR will predict properly.

        predict = coarse_predict.clone().detach()
        predict[predict==2] = 4

        NPDR_inds_subset = torch.nonzero(coarse_predict == 1).squeeze(1)  # for multi label... how do we do?
        if len(NPDR_inds_subset) > 0:
            fine_predict = pred[NPDR_inds_subset,3:6].argmax(dim=1)
            fine_predict += 1  # todo use reg value to analyze. but need to know the threshold

            predict[NPDR_inds_subset] = fine_predict

        return predict


def quadratic_kappa(y_hat, y):
    # need to convert our answer format
    y_hat = convert_to_normal_pred(y_hat)
    y = y[:,-1].int()

    #cohen_kappa_score(y_hat, y,  weights='quadratic')
    #cohen_kappa_score(y_hat, y, labels=['normal', 'NPDR_1', 'NPDR_2', 'NPDR_3', 'PDR'], weights='quadratic')
    return torch.tensor(cohen_kappa_score(y_hat, y, weights='quadratic'), device='cuda:0')


# **Training:**
#
# We use transfer learning, where we retrain the last layers of a pretrained neural network. I use the ResNet50 architecture trained on the ImageNet dataset, which has been commonly used for pre-training applications in computer vision. Fastai makes it quite simple to create a model and train:



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
        head = create_head(nf, data.c+2, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    else:
        head = custom_head

    model = nn.Sequential(body, head)

    learn = Learner(data, model, **kwargs)
    learn.split(split_on or meta['split'])
    if pretrained: learn.freeze()
    if init: apply_init(model[1], init)
    return learn


# modified based https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class DR_FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0., alpha=0.5, eps=1e-7, magnifier=1., from_logits=True):
        super(DR_FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.magnifier = magnifier
        assert self.alpha >= 0
        assert self.alpha <= 1
        assert self.magnifier > 0
        self.from_logits = from_logits

        self.a_normal = (3662 - 1805) / 3662
        self.a_NPDR = (3662 - 1562) / 3662
        self.a_PDR = (3622 - 295) / 3662
        # (370, 999, 193)
        self.a_NPDR_1 = 1 - 370/1562
        self.a_NPDR_2 = 1 - 999/1562
        self.a_NPDR_3 = 1 - 193/1562

        self.coarse_mag = 1.
        self.fine_mag = 1.5
        self.reg_mag = 0.5

        self.coarse_a = [self.a_normal, self.a_NPDR, self.a_PDR]
        self.fine_a = [self.a_NPDR_1, self.a_NPDR_2, self.a_NPDR_3]
        #self.device = torch.device('cuda:0')
        #self.coarse_a = torch.tensor([self.a_normal, self.a_NPDR, self.a_PDR], device=self.device)
        #self.fine_a = torch.tensor([self.a_NPDR_1, self.a_NPDR_2, self.a_NPDR_3], device=self.device)

    def forward(self, input, target, **kwargs):
        # Starting var:.. input = tensor<(64, 5), float32, cuda:0, grad>
        # Starting var:.. target = tensor<(64,), int64, cuda:0>

        # so for every [,:] data, 5 -> 1, we set the loss
        # for [,:0] [,:4] we use binary classification, [,:1-3] too. for [,:1~3], we do regression too


        # now we have this: [,:0~2] [,:2~5] use the binary CE, for [:, 6] use regression
        # so we have 6 logits and one tensor for regression
        reduction_param = kwargs.get('reduction', 'mean')
        target = target.float()  # otherwise, int*float thing

        reg_score_pred = input[:, -1]
        if self.from_logits:
            cls_input = torch.sigmoid(input[:, :-1])
        else:
            cls_input = input[:, :-1]

        coarse_cls_input = cls_input[:,:3]
        coarse_target = target[:, :3]

        # todo add loss/constraint for NPDR,PDR?
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
            loss += fine_cls_loss

        loss += self.reg_mag * F.smooth_l1_loss(reg_score_pred, target[:, -1], reduction=reduction_param)

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
        loss = -y.new_tensor(alpha) *        pos_gamma_balancer * y * torch.log(y_pred)  # cross entropy
        loss += -(1-y.new_tensor(alpha)) * neg_gamma_balancer * not_y * torch.log(not_y_pred)
        loss *= mag

        #if target.requires_grad:  # we don't care about this, so just ignore
        
        if reduction != 'none':
            return torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        #else:
        #    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        #    ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
        #return ret
        return loss    
    


fl_normal = DR_FocalLoss(gamma=0., alpha=0.5, magnifier=1.)  # changed lr decay 2/0.15 + patience=3, do not use focal loss...
# set_trace() we convert back to ipynb
# learner = DR_learner(data, vision.models.densenet121, metrics=[quadratic_kappa])
learner = DR_learner(data, vision.models.densenet121, loss_func=fl_normal, metrics=[quadratic_kappa])

#learn = cnn_learner(data, base_arch=models.resnet50, metrics=[quadratic_kappa])
learn = learner


# +
#savedPath = Path('models/stage-2.pth')
#if savedPath.exists():
#if False:
#    learn.load('stage-2')
#else:
learn.lr_find()
learn.recorder.plot(suggestion=True)
# LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
# Min numerical gradient: 3.31E-02
# Min loss divided by 10: 2.09E-02

# after we change our data formation...
#LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
#Min numerical gradient: 1.74E-03
#Min loss divided by 10: 1.20E-02

# +
# Here we can see that the loss decreases fastest around `lr=1e-2` so that is what we will use to train:



learn.fit_one_cycle(4, max_lr=2e-3)
# -

learn.recorder.plot_losses()
learn.recorder.plot_metrics()

learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

# +
# LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
# Min numerical gradient: 1.91E-06
# Min loss divided by 10: 3.31E-07



learn.fit_one_cycle(6, max_lr=slice(1e-6, 2e-3/5))
# -

learn.recorder.plot_losses()
learn.recorder.plot_metrics()

learn.export()
learn.save('stage-2')

# Let's evaluate our model:
pdb.run('interp = ClassificationInterpretation.from_learner(learn)')

#  b /opt/conda/lib/python3.6/site-packages/fastai/basic_train.py:43
interp = ClassificationInterpretation.from_learner(learn)

pdb.pm()

# +
losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)
# -



interp.preds[idxs[:20]]

interp.plot_top_losses(k=12, heatmap=False)

pdb.run('interp.plot_top_losses(k=9)')
# /opt/conda/lib/python3.6/site-packages/fastai/vision/learner.py:147
# /opt/conda/lib/python3.6/site-packages/fastai/vision/learner.py

pdb.pm()

# !cat -n /opt/conda/lib/python3.6/site-packages/fastai/data_block.py

interp.plot_confusion_matrix(figsize=(12,12), dpi=98)

# +
# ## Optimize the Metric
#
# Optimizing the quadratic kappa metric was an important part of the top solutions in the previous competition. Thankfully, @abhishek has already provided code to do this for us. We will use this to improve the score.
valid_preds = learn.get_preds(ds_type=DatasetType.Valid)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
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

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
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
        return X_p

    def coefficients(self):
        return self.coef_['x']

#optR = OptimizedRounder()
#optR.fit(valid_preds[0], valid_preds[1])

#coefficients = optR.coefficients()
#print(coefficients)
# -

valid_preds[1].shape


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

def _TTA(learn: Learner, beta: float = 0, ds_type: DatasetType = DatasetType.Valid, num_pred: int = 10,
         with_loss: bool = False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds, y = learn.get_preds(ds_type)
    all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None:
        return preds, avg_preds, y
    else:
        final_preds = preds * beta + avg_preds * (1 - beta)
        if with_loss:
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss
        return final_preds, y


Learner.TTA = _TTA

sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()
learn.data.add_test(
    ImageList.from_df(sample_df, '../input/aptos2019-blindness-detection', folder='test_images', suffix='.png'))

preds, y = learn.TTA(ds_type=DatasetType.Test)
test_predictions = convert_to_normal_pred(preds)
#test_predictions = optR.predict(preds, coefficients)

sample_df.diagnosis = test_predictions.astype(int)
sample_df.head()

sample_df.to_csv('submission.csv', index=False)
# -