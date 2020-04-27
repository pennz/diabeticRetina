```
[[355   6   0   0   0]
 [  7  53  11   3   0]
 [  0  38 132  14  16]
 [  0   1   6  10  22]
 [  0   7  15   4  33]]
```
this is the one use argmax thing to predict and PDR logit + reg (and use
resnet 50) (almost the same for the submitted one)
got value 0.8764525281210962

```
# after change reg threshold for PDR. 0.8822022272803653(thresh 3.38), thresh 3 0.8854813965563013
array([[355,   6,   0,   0,   0],
       [  7,  53,  11,   3,   0],
       [  0,  38, 132,  30,   0],
       [  0,   1,   6,  32,   0],
       [  0,   7,  15,  26,  11]])

# below thresh 3 (smaller then last 3.38) (score 0.88)
array([[355,   6,   0,   0,   0],
       [  7,  53,  11,   3,   0],
       [  0,  38, 132,  29,   1],
       [  0,   1,   6,  23,   9],
       [  0,   7,  15,  17,  20]])


# this only use the regression value. so for NPDR3, reg is better
# got value 0.90 .... use the optimized kappa split points (not optimized 0.898)
array([[353,   8,   0,   0,   0],
       [  7,  40,  23,   4,   0],
       [  0,  15, 128,  57,   0],
       [  0,   1,   4,  34,   0],
       [  0,   0,  11,  37,  11]])
```



(For prediction ) NPDR1, use reg, NPDR2, use argmax thing, 



```
0,1.156973,1.006757,0.678325,12:23
1,0.737687,0.484378,0.843250,12:16
2,0.572041,0.424307,0.875032,12:16
epoch,train_loss,valid_loss,quadratic_kappa,time
0,0.520466,#na#,10:22
1,1.566402,#na#,05:44
epoch,train_loss,valid_loss,quadratic_kappa,time
0,0.485927,0.410838,0.870430,12:25
1,0.481519,0.393758,0.864738,12:21
2,0.468695,0.392120,0.868664,12:22
```
use densenet 121 to predict

```
epoch 	train_loss 	valid_loss 	quadratic_kappa 	time
0 	0.944560 	0.847292 	0.660102 	09:47
1 	0.694452 	0.673022 	0.821088 	09:42
2 	0.553161 	0.436186 	0.861933 	09:41

epoch   train_loss  valid_loss  quadratic_kappa     time
0   0.467912    0.425189    0.877827    09:44
1   0.455554    0.390629    0.873190    09:42
2   0.438599    0.386885    0.869910    09:44<Paste>
```
And this is resnet 50. check the 'res50-result.ipynb', could train longer


# results

另外，我的pulic test set的预测结果，说明test里面很多NPDR2,NPDR3的，以及PDR的。
所以之后数据往这几类多填充填充

todo : 
- if you have resized images on local computer make sure you do same resizing when you use Kaggle kernel for submission 
- follow and reimplement what the winner got
- write testsuit for automation thing
# just check the great one, and try to replicate it

my previous resutl:
0.723, need to improve. Previous direction is not right.
for quick loopback, edit locally and separate well, for better tesing (with local configuration maybe)
set mail stone, and check if it is done
