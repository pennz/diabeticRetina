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
1. more data training
2. cv2 preprocessing