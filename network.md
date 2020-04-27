  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): None
  (last_linear): Sequential(
    (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Dropout(p=0.25)
    (2): Linear(in_features=2048, out_features=2048, bias=True)
    (3): ReLU()
    (4): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=2048, out_features=1, bias=True)
  __)
  (avg_pool): AdaptiveAvgPool2d(output_size=1)
)


# 是否需要设计网络来更新分类权重？
回归自身有个loss，分类有个loss，直接加起来？
回归的话，是

# todo
1. more data training (balance different class) (for crop, use cropped ones, just do more cropping)
2. cv2 preprocessing, remove the black part
3. data norm thing

## trials

### + data mean for DR data
4 freezed training

epoch 	train_loss 	valid_loss 	quadratic_kappa 	time
0 	4.714076 	6.048602 	0.229955 	09:36
1 	3.571989 	3.295304 	0.690442 	09:25
2 	2.915984 	2.985442 	0.791941 	09:28
3 	2.374688 	1.966715 	0.850125 	09:28

### repaired the loss function
epoch,train_loss,valid_loss,quadratic_kappa,time
0,6.090740,3.543248,0.781348,09:57 
1,4.097026,2.938734,0.811764,09:37
2,3.227793,2.403766,0.881219,09:29 
3,2.720900,2.331233,0.874537,09:20 

Local CV seems better.

## hypothesis
batch norm, remember the stats, so run longer and it is better?

