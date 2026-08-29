
```
| 标           | SimpleCNN |   ResNet18 |    变化 |
| ------------ | --------: | ---------: | -----: |
| Best Val Acc |    97.75% | **97.95%** | +0.20% |
| Test Acc     |    88.15% | **91.15%** | +3.00% |
| Macro-F1     |    87.64% | **90.08%** | +2.44% |
| Test Loss    |    0.4654 | **0.4069** |    更低 |


```



```

ResNet18 Baseline V1

Parameters      11,173,323
Best Epoch      7
Best Val Acc    97.95%
Test Acc        91.15%
Macro-F1        90.08%

现象：
明显过拟合，
但相比 SimpleCNN，
Test Accuracy 和 Macro-F1 均明显提高。

```