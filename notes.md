# 相关医学知识
[diabatic knowledge (Chinese)](https://zhuanlan.zhihu.com/p/28954163)
## 什么是糖尿病视网膜病变

糖尿病对微血管的破坏而导致的视网膜病变
- 微血管由于糖尿病被破坏（血液中血糖之类化学物质成分变化，使得视网膜微血管病变）
- 微血管破坏后，视网膜缺氧，缺血
- 进而，视网膜微血管瘤形成、出血、滲出、玻璃体出血、新生血管形成及玻璃体视网膜增殖性改变等
- 以上统称为糖尿病视网膜病变

## 分类与病程
[classification, NPDR v.s. PDR](https://webeye.ophth.uiowa.edu/eyeforum/tutorials/diabetic-retinopathy-med-students/Classification.htm)
糖尿病视网膜病变可分为NPDR和PDR，即 非增殖性糖尿病视网膜病变 与 增殖性糖尿病视网膜病变。

NPDR（非增殖性糖尿病视网膜病变）细分为3类:

- Early NPDR 

    At least one microaneurysm（微血管瘤） present on retinal exam.
- Moderate NPDR

    Characterized by multiple microaneurysms, dot-and-blot hemorrhages（斑点状出血、滲出）, venous beading, and/or cotton wool spots（静脉珠状、絮状斑点）.
- Severe NPDR

    In the most severe stage of NPDR, you will find cotton wool spots, venous beading, and severe intraretinal microvascular abnormalities (IRMA，视网膜内微血管异常). 
    It is diagnosed using the "4-2-1 rule." 
    A diagnosis is made if the patient has any of the following: 
    - diffuse intraretinal hemorrhages and microaneurysms(弥漫性视网膜内出血和微动脉瘤) in 4 quadrants, 
    - venous beading (静脉珠状点) in ≥2 quadrants, or 
    - IRMA (视网膜内微血管异常) in ≥1 quadrant. 
    
    Within one year, 52-75% of patients falling into this category will progress to PDR (Aiello 2003).

As the disease progresses, it may evolve into proliferative diabetic retinopathy (PDR), which is defined by the presence of neovascularization(新增殖的血管) and has a greater potential for serious visual consequences.

可以看到，NPDR这一分类是指的kaggle数据里面的1～3类。NPDR有可能发展成为PDR（增殖性糖尿病视网膜病变），即类4。下图为一例：

![NPDR example](https://www.columbiaeye.org/sites/default/files/pictures/nveb.jpg)

[kaggle page](https://www.kaggle.com/c/aptos2019-blindness-detection)

[National Eye institute page](https://nei.nih.gov/health/diabetic/retinopathy)

## Proliferative Diabetic Retinopathy (PDR) and NPDR
[American College of Physicians](https://www.acponline.org/meetings-courses/internal-medicine-meeting/ophthalmology-self-guided-study-activity-herbert-s-waxman-clinical-skills-center/proliferative-diabetic-retinopathy-pdr)

[Stages of DR](http://www.youreyes.org/eyehealth/diabetic-retinopathy)

Four stages: mild->moderate->severe->proliferactive

# 1st place solution
## Validation Strategy
I made the decision to combine the whole 2015 and 2019 data as train set, and solely relied on public LB for validation.
## Preprocessing
no need, resolution is fine
## Models and input sizes
2 x inception_resnet_v2, input size 512
2 x inception_v4, input size 512
2 x seresnext50, input size 512
2 x seresnext101, input size 384
*Inceptions and ResNets usually blend well. If I could have two more weeks I would definitely add some EfficientNets. *

The input size was mainly determined by observations in the 2015 competition that larger input size brought better performance. 
## Loss, Augmentations, Pooling
I used only nn.SmoothL1Loss() as the loss function. Other loss functions may work well too. I sticked to this single loss just to simplify the emsembling process.
```
contrast_range=0.2,
brightness_range=20.,
hue_range=10.,
saturation_range=20.,
blur_and_sharpen=True,
rotate_range=180.,
scale_range=0.2,
shear_range=0.2,
shift_range=0.2,
do_mirror=True,
```
For the last pooling layer, I found the generalized mean pooling (https://arxiv.org/pdf/1711.02512.pdf) better than the original average pooling. Code copied from https://github.com/filipradenovic/cnnimageretrieval-pytorch.

## Training and Testing
### two stage training
- routinely trained the eight models and validated each of them on the public LB (To get more stable results, models were evaluated in pairs (with different seeds), that's why I have 2x for each type of model. )(When probing LB, I tried to reduce the degree of freedom of hyperparemeters to alleviate overfitting, for example, to determine the best number of epochs for training I used a step size of five. )
- In the second stage of training, I added pseudo-labelled (soft version) public test data and two additional external data - the Idrid and the Messidor dataset, to the stage1 trainset.
