# OrganAMNIST 医学图像分类：第一个完整 PyTorch CNN 项目

这是我学习 PyTorch 过程中完成的第一个相对完整的 CNN 图像分类项目。

项目使用 **OrganAMNIST** 数据集，实现了一个简单的卷积神经网络（SimpleCNN），并完整走通了：

```text
数据加载
→ CNN 建模
→ Batch 训练
→ Epoch 训练
→ Validation
→ Dropout
→ 保存最佳模型
→ Test
→ Accuracy
→ Macro-F1
→ Classification Report
→ Confusion Matrix
→ 项目模块化
```

这个项目主要用于学习，因此 README 会如实记录实验结果，也会记录过程中出现的问题和我的理解，而不只保留“最好看的结果”。

---

## 1. 项目目的

一开始做这个项目，主要是为了真正理解一个 CNN 是怎么从数据一直训练到最终评价的，而不是只会调用现成模型。

在这个过程中逐步理解了：

* Dataset 和 DataLoader
* Sample、Batch、Epoch 的关系
* Conv、ReLU、Pooling、FC
* Forward Propagation
* CrossEntropyLoss
* Gradient 和 Backpropagation
* Optimizer
* Train / Validation / Test 的区别
* `model.train()` 和 `model.eval()`
* Dropout
* Overfitting
* Best Checkpoint
* Accuracy
* Precision / Recall / F1-score
* Macro-F1
* Confusion Matrix
* PyTorch 项目模块化

---

## 2. 数据集

本项目使用 MedMNIST 中的 **OrganAMNIST**。

数据划分：

| 数据集        |   样本数量 |
| ---------- | -----: |
| Train      | 34,561 |
| Validation |  6,491 |
| Test       | 17,778 |

图像为：

```text
1 × 28 × 28
```

即单通道灰度图像。

分类任务共包含 11 个类别：

```text
bladder
femur-left
femur-right
heart
kidney-left
kidney-right
liver
lung-left
lung-right
pancreas
spleen
```

---

## 3. CNN 模型

当前使用的是自己搭建的 SimpleCNN，而不是直接调用 ResNet 等成熟网络。

基本结构：

```text
Input
[1, 28, 28]
      ↓
Conv2d
1 → 32
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2d
32 → 64
      ↓
ReLU
      ↓
MaxPool
      ↓
Flatten
      ↓
FC
3136 → 128
      ↓
ReLU
      ↓
Dropout
      ↓
FC
128 → 11
      ↓
11 类预测
```

加入 Dropout 的主要目的是减轻过拟合。

---

## 4. 项目结构

最开始训练、验证、测试等代码全部写在 `main.py` 中。

随着项目逐渐完整，后来将代码拆分为：

```text
PyTorch_pra/
│
├── LICENSE
│
├── .gitignore
│
├── common/                      ← 共享工具包（供 simple_cnn、resnet18 等复用）
│   ├── __init__.py
│   ├── data.py
│   │   └── OrganAMNIST 数据集和 DataLoader
│   └── engine.py
│       ├── train_one_epoch()
│       ├── evaluate()
│       └── predict()
│
├── simple_cnn/
│   ├── model.py
│   │   └── SimpleCNN 模型
│   ├── main.py
│   │   └── 组织完整实验流程（入口处自动注入项目根到 sys.path 导入 common）
│   ├── best_model.pth
│   │   └── 验证集表现最好的模型（运行 main.py 后生成）
│   └── README.md
│
└── resnet18/
    ├── __init__.py
    ├── basic_block.py
    ├── resnet18.py
    └── main.py
```

现在各文件/包的职责比较明确：

```text
common/data.py
负责数据（可被 simple_cnn、resnet18 复用）

common/engine.py
负责训练和评价逻辑（可被 simple_cnn、resnet18 复用）

simple_cnn/model.py
负责 SimpleCNN 模型

simple_cnn/main.py
负责把整个实验组织起来，并在入口注入项目根目录，保证跨目录导入 common/ 稳定
```

这也是我第一次真正理解代码“模块化”是干什么的。

---

## 5. 训练流程

当前主要训练配置：

```text
Batch Size     = 64
Epoch          = 10
Optimizer      = Adam
Learning Rate  = 0.001
Loss           = CrossEntropyLoss
Regularization = Dropout
Device         = Apple MPS（可用时）
```

一个 Epoch 内部的训练过程：

```text
取一个 Batch
      ↓
optimizer.zero_grad()
      ↓
Forward
      ↓
计算 Loss
      ↓
loss.backward()
      ↓
计算梯度
      ↓
optimizer.step()
      ↓
更新模型参数
```

每完成一个 Epoch 后，在 Validation Set 上进行一次评价。

```text
Train
   ↓
Validation
   ↓
比较 Val Accuracy
   ↓
如果当前最好
   ↓
保存 best_model.pth
```

所有 Epoch 完成以后：

```text
加载 best_model.pth
        ↓
Test Set
        ↓
最终模型评价
```

---

## 6. 最终一次实验结果

本次实验中，最佳 Validation 模型出现在：

```text
Epoch 7
Val Accuracy = 0.9775
```

训练日志：

```text
Epoch 1/10
Train Loss: 0.8036
Train Acc : 0.7193
Val Loss  : 0.1573
Val Acc   : 0.9590

Epoch 2/10
Train Loss: 0.3609
Train Acc : 0.8727
Val Loss  : 0.1063
Val Acc   : 0.9715

Epoch 3/10
Train Loss: 0.2638
Train Acc : 0.9090
Val Loss  : 0.1006
Val Acc   : 0.9669

Epoch 4/10
Train Loss: 0.2003
Train Acc : 0.9299
Val Loss  : 0.1312
Val Acc   : 0.9698

Epoch 5/10
Train Loss: 0.1643
Train Acc : 0.9428
Val Loss  : 0.1446
Val Acc   : 0.9656

Epoch 6/10
Train Loss: 0.1376
Train Acc : 0.9514
Val Loss  : 0.1521
Val Acc   : 0.9686

Epoch 7/10
Train Loss: 0.1184
Train Acc : 0.9582
Val Loss  : 0.0986
Val Acc   : 0.9775

Epoch 8/10
Train Loss: 0.1017
Train Acc : 0.9635
Val Loss  : 0.1171
Val Acc   : 0.9744

Epoch 9/10
Train Loss: 0.0908
Train Acc : 0.9678
Val Loss  : 0.1609
Val Acc   : 0.9770

Epoch 10/10
Train Loss: 0.0821
Train Acc : 0.9703
Val Loss  : 0.1193
Val Acc   : 0.9766
```

最终加载 Epoch 7 的最佳模型，在 Test Set 上得到：

| 指标                |         结果 |
| ----------------- | ---------: |
| Best Val Accuracy | **97.75%** |
| Test Accuracy     | **88.15%** |
| Test Macro-F1     | **87.64%** |
| Test Loss         | **0.4654** |

---

## 7. 一个值得注意的问题：Val 和 Test 差距较大

这次实验中：

```text
Best Val Accuracy = 97.75%

Test Accuracy     = 88.15%
```

两者存在约 9.6 个百分点的差距。

这个结果不会为了让实验“好看”而隐藏。

目前可以确认的是：

* 最终 Test 使用的是独立 `test_loader`
* Test 前重新加载了 Validation Accuracy 最佳的模型
* 最佳模型来自 Epoch 7
* Test Accuracy 仍明显低于 Validation Accuracy

因此目前更合理的结论是：

> 当前 SimpleCNN 在 Validation Set 上表现很好，但在 Test Set 上的泛化性能明显下降。

造成这一现象的具体原因还没有在本项目中进一步深入研究。

这个问题也会作为以后使用更强模型时的重要对照。

---

## 8. Dropout 后出现的一个有意思现象

加入 Dropout 后，一段时间内出现：

```text
Train Accuracy < Validation Accuracy
```

最开始我以为这是代码出现了问题。

后来理解到：

训练阶段：

```python
model.train()
```

Dropout 开启，会随机关闭部分特征，因此训练时的预测任务实际上更困难。

Validation 阶段：

```python
model.eval()
```

Dropout 被关闭，完整网络参与预测。

同时 Train Accuracy 是在整个 Epoch 一边更新模型、一边累计得到的，而 Validation Accuracy 使用的是该 Epoch 训练结束后的模型。

因此：

```text
Val Accuracy > Train Accuracy
```

在这里并不意味着程序一定有错误。

---

## 9. 训练过程中出现过的问题

这个项目不是一次写对的，中间出现过不少问题。

### 9.1 一开始只训练了 100 个 Batch

最开始只跑了：

```text
Batch 10
Batch 20
...
Batch 100
```

当时还不理解为什么 Loss 曲线一直抖动。

后来才真正理解：

```text
1 sample
    ↓
64 samples = 1 batch
    ↓
约 541 batches = 1 epoch
    ↓
多个 epochs = 多次遍历整个训练集
```

Batch Loss 本身就容易波动，Epoch 平均 Loss 更适合观察整体趋势。

---

### 9.2 一开始不理解完整训练循环

后来把 PyTorch 最核心的训练过程总结成：

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

对应：

```text
清空梯度
→ 前向传播
→ 计算损失
→ 反向传播
→ 更新参数
```

这几行也是整个项目中最重要的基础。

---

### 9.3 曾经出现 Val 和 Test 结果异常一致

实验过程中曾出现：

```text
Val Loss == Test Loss
Val Accuracy == Test Accuracy
```

而且两者几乎完全相同。

由于：

```text
Validation samples = 6,491
Test samples       = 17,778
```

这种结果非常可疑。

因此重新检查了：

```text
val_dataset
test_dataset
val_loader
test_loader
Test loop
```

修正数据/测试流程后，Test 结果恢复为与 Validation 明显不同的正常结果。

这次问题让我意识到：

> 模型结果“看起来很好”不代表代码一定正确。

当结果异常漂亮或者异常巧合时，首先应该检查数据和评价流程。

---

### 9.4 最开始保存的 best model 不知道来自哪个 Epoch

最初只保存：

```python
torch.save(
    model.state_dict(),
    "best_model.pth"
)
```

虽然模型参数保存成功了，但看不出来它到底来自：

```text
Epoch 2？
Epoch 3？
Epoch 7？
```

后来增加：

```python
best_epoch
best_val_accuracy
```

现在训练时可以明确看到：

```text
保存当前最佳模型：
Epoch 7
Val Acc = 0.9775
```

也让我真正理解了 Best Checkpoint 的意义：

> 最后一个 Epoch 不一定是最好的模型。

---

### 9.5 出现过拟合趋势

随着 Epoch 增加：

```text
Train Loss 持续下降
Train Accuracy 持续提高
```

但 Validation 并没有一直同步改善。

这让我第一次通过自己的实验理解：

```text
训练集越来越好
≠
模型泛化能力一定越来越好
```

因此后来加入：

```text
Dropout
+
Validation
+
Best Checkpoint
```

来控制和观察这一问题。

---

## 10. Test 分类结果

最终：

```text
Accuracy  = 0.8815
Macro-F1 = 0.8764
```

每个类别的 F1-score：

| Class        |   F1-score |
| ------------ | ---------: |
| bladder      |     0.8077 |
| femur-left   |     0.8997 |
| femur-right  |     0.8907 |
| heart        |     0.8704 |
| kidney-left  | **0.7184** |
| kidney-right |     0.8103 |
| liver        | **0.9784** |
| lung-left    | **0.9898** |
| lung-right   | **0.9838** |
| pancreas     |     0.8838 |
| spleen       |     0.8074 |

可以看到模型对不同器官的识别能力并不一致。

表现最好的是：

```text
lung-left
lung-right
liver
```



相对最困难的是：

```text
kidney-left
```

---

## 11. 为什么不能只看 Accuracy

这个实验最终还加入了：

```text
Accuracy
Macro-F1
Classification Report
Confusion Matrix
```

它们回答的问题不同。

### Accuracy

回答：

> 17,778 张 Test 图片中，总共有多少比例预测正确？

本实验：

```text
Accuracy = 88.15%
```

### Macro-F1

先分别计算 11 个类别的 F1-score，再让 11 个类别拥有相同权重进行平均。

本实验：

```text
Macro-F1 = 87.64%
```

Accuracy 和 Macro-F1 比较接近，说明当前结果并不是完全依靠某几个样本数量较大的类别撑起来的。

### Confusion Matrix

Confusion Matrix 更像模型的“错题本”。

它不仅告诉我模型预测错了，还能告诉我：

> 真实是什么，被模型错认成了什么？

例如本次实验比较明显的错误包括：

```text
kidney-right → kidney-left    319
spleen       → kidney-left    274
kidney-left  → spleen         138
kidney-left  → bladder        109
kidney-left  → pancreas        80
```

因此可以进一步发现：

> 当前 SimpleCNN 对 kidney-left、kidney-right、spleen 等类别的区分能力相对较弱。

这也是只看 Accuracy 无法发现的信息。

---

## 12. 我目前对模型评价的理解

完成这个项目以后，我目前把分类模型评价理解成：

```text
Loss Curve
    ↓
模型有没有正常收敛？
有没有出现过拟合？

Accuracy
    ↓
总体预测正确率是多少？

Macro-F1
    ↓
各个类别是否整体都表现得比较好？

Classification Report
    ↓
每个类别的 Precision / Recall / F1 是多少？

Confusion Matrix
    ↓
模型具体把谁错认成了谁？
```

这些指标不是互相替代，而是在从不同角度观察同一个模型。

---

## 13. 运行项目

安装主要依赖：

```bash
pip install torch torchvision medmnist matplotlib scikit-learn
```

运行：

```bash
cd simple_cnn
python main.py
```

程序会依次完成：

```text
加载 OrganAMNIST
        ↓
创建 DataLoader
        ↓
创建 SimpleCNN
        ↓
训练
        ↓
每个 Epoch 做 Validation
        ↓
保存最佳模型
        ↓
加载最佳模型
        ↓
Test
        ↓
Accuracy
        ↓
Macro-F1
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
Loss / Accuracy Curve
```

---

## 14. 目前项目的不足

这个项目目前只是一个学习性质的 CNN Baseline，还有很多没有做的内容。

### 14.1 模型比较简单

目前使用的 `SimpleCNN` 只有两层卷积：

```text
Conv1
↓
ReLU
↓
MaxPool
↓
Conv2
↓
ReLU
↓
MaxPool
↓
FC
↓
Dropout
↓
FC
```

这个结构主要用于理解 CNN 和完整训练流程，并不是为了追求 OrganAMNIST 上的最佳性能。

目前还没有使用：

* ResNet 等更深的网络
* Batch Normalization
* Learning Rate Scheduler
* 更系统的数据增强
* Transfer Learning
* 更复杂的正则化方法

这些内容暂时不加入，是因为当前阶段的主要目标是先把最基本的 CNN 和 PyTorch 训练流程真正弄明白。

---

### 14.2 Validation 和 Test 的差距比较明显

本次实验中：

```text
Best Validation Accuracy = 97.75%
Test Accuracy            = 88.15%
```

两者相差约：

```text
9.60 个百分点
```

这个差距值得注意。

目前还没有进一步系统分析造成这个现象的原因，因此不能简单地把它归结为某一个因素。

后续可以进一步检查：

* Validation 与 Test 数据分布是否存在差异
* 不同类别在两个数据集中的分布情况
* 模型是否对 Validation Set 存在一定程度的适应
* 不同随机初始化下结果是否稳定
* 更强模型能否缩小这一差距

因此，目前的 `88.15% Test Accuracy` 应当看作这个 SimpleCNN Baseline 在本次实验中的实际测试结果，而不能只根据较高的 Validation Accuracy 判断模型性能。

---

### 14.3 不同类别的分类能力并不一致

虽然整体：

```text
Test Accuracy = 88.15%
Macro-F1      = 87.64%
```

但从 Classification Report 和 Confusion Matrix 可以发现，不同类别之间仍然存在明显差异。

例如：

```text
lung-left      F1 = 0.9898
lung-right     F1 = 0.9838
liver          F1 = 0.9784
```

这些类别表现很好。

但是：

```text
kidney-left    F1 = 0.7184
```

明显较低。

混淆矩阵进一步发现：

```text
kidney-right → kidney-left
spleen       → kidney-left
kidney-left  → spleen
```

等错误相对较多。

因此，单独看 Accuracy 会遗漏这些类别层面的错误。

这也是这个项目后期加入 Macro-F1、Classification Report 和 Confusion Matrix 的原因。

---

### 14.4 目前只进行了一次完整实验

目前 README 中记录的结果来自一次完整训练：

```text
10 Epochs
Batch Size = 64
Learning Rate = 0.001
Adam
Dropout
```

但是神经网络存在：

* 参数随机初始化
* DataLoader shuffle
* Dropout 随机失活

因此重新运行程序，结果不会保证完全一样。

目前还没有加入：

```python
torch.manual_seed(...)
```

等随机种子控制，也没有进行多次独立实验并计算：

```text
Mean ± Standard Deviation
```

所以当前结果适合作为学习和 Baseline 记录，但还不能当成严格的重复实验结果。

---

### 14.5 超参数还没有系统调优

目前使用：

```text
batch_size = 64
learning_rate = 0.001
num_epochs = 10
optimizer = Adam
dropout = 0.5
```

这些参数主要用于完成学习实验。

目前没有系统比较：

```text
batch_size = 32 / 64 / 128

learning_rate =
0.01 / 0.001 / 0.0001

dropout =
0.2 / 0.3 / 0.5
```

也没有为了提高测试集成绩反复调参。

这一点是有意保留的。

当前项目的目标是理解训练流程，而不是通过大量调参把 OrganAMNIST 的测试结果做到最高。

---

## 15. 开发过程中遇到的问题

这个项目并不是一次写完的，中间出现过不少问题。这里把它们保留下来，作为学习记录。

### 15.1 一开始只训练了 100 个 Batch

最初的训练代码只运行：

```text
Batch 1
Batch 2
...
Batch 100
```

当时还没有真正理解 Epoch。

后来才明确：

```text
1 sample
↓
多个 sample 组成 1 batch
↓
整个训练集所有 batch 跑完一次
↓
1 epoch
```

OrganAMNIST 训练集共有：

```text
34,561 张图片
```

当：

```text
batch_size = 64
```

时，一个 Epoch 大约包含 541 个 Batch。

随后训练代码才从“跑若干 Batch”改成了完整的 Epoch Training Loop。

---

### 15.2 最开始的 Loss 曲线抖动很明显

一开始记录的是：

```text
每个 Batch 的 Loss
```

因此曲线存在明显波动。

后来改为统计：

```text
一个 Epoch 内所有样本的平均 Loss
```

曲线才更适合观察整体训练趋势。

这也让我理解了：

> Batch Loss 的局部波动并不一定说明训练出了问题，更重要的是观察整体趋势。

---

### 15.3 一开始只会训练，不知道为什么还需要 Validation

最初只有：

```text
Train
```

后来逐渐加入：

```text
Train
↓
Validation
↓
Test
```

并理解三者的区别：

```text
Train
用于学习参数

Validation
用于观察模型泛化表现和选择模型

Test
用于最后评价模型
```

Test 不应该参与模型训练，也不应该被反复用来选择最佳 Epoch。

---

### 15.4 曾经出现 Validation 和 Test 结果完全一样的问题

开发过程中曾经得到过：

```text
Val Loss = 0.0780
Val Acc  = 0.9731

Test Loss = 0.0780
Test Acc  = 0.9731
```

由于 Validation Set 和 Test Set 的样本数量不同，这个结果非常可疑。

于是重新检查了：

```text
val_loader
test_loader
```

以及 Test 部分的代码。

修正后得到的 Test Accuracy 明显低于 Validation Accuracy，结果恢复正常。

这次问题让我意识到：

> 当实验结果“好得过于巧合”时，应该先检查代码和数据流程，而不是马上相信结果。

---

### 15.5 最开始直接使用最后一个 Epoch 做 Test

早期版本训练完成以后，直接使用：

```text
Epoch 10
```

的模型进行 Test。

后来发现：

```text
最后一个 Epoch
≠
Validation 表现最好的 Epoch
```

因此加入 Best Checkpoint：

```python
if val_accuracy > best_val_accuracy:
    torch.save(...)
```

最终流程变成：

```text
Train
↓
Validation
↓
发现更好的模型
↓
保存
↓
全部 Epoch 完成
↓
重新加载 Best Model
↓
Test
```

本次实验最终保存的是：

```text
Epoch 7
Val Accuracy = 97.75%
```

---

### 15.6 一开始不知道保存的到底是哪一个 Epoch

最初保存模型时只打印：

```text
保存当前最佳模型
```

因此虽然程序确实保存了最佳模型，但无法直观看出它来自哪个 Epoch。

后来增加：

```python
best_epoch = epoch + 1
```

最终可以直接输出：

```text
保存当前最佳模型：Epoch 7, Val Acc: 0.9775
```

这样模型选择过程更加清楚。

---

### 15.7 出现了轻微过拟合，因此加入 Dropout

随着 Epoch 增加，曾观察到：

```text
Train Loss 继续下降
Train Accuracy 继续提高

但是

Validation Loss 不再稳定下降
Validation Accuracy 基本不再提高
```

因此开始接触 Overfitting，并在全连接层加入：

```python
nn.Dropout(p=0.5)
```

加入 Dropout 后也进一步理解了：

```python
model.train()
```

和：

```python
model.eval()
```

并不只是形式上的写法。

训练模式下：

```text
Dropout 开启
```

验证和测试模式下：

```text
Dropout 关闭
```

---

### 15.8 Train Accuracy 一度明显低于 Validation Accuracy

加入 Dropout 后曾出现：

```text
Train Accuracy < Validation Accuracy
```

一开始以为代码存在问题。

后来理解到主要有两个原因：

一是训练阶段：

```text
model.train()
→ Dropout 开启
```

模型实际上是在更困难的情况下进行预测。

二是 Train Accuracy 是在一个 Epoch 中：

```text
边训练
边更新参数
边统计
```

而 Validation 是在整个 Epoch 训练结束后，使用当前模型统一计算。

因此这种情况并不一定表示程序错误。

---

### 15.9 原来的 main.py 越写越长

项目最初几乎所有代码都放在：

```text
main.py
```

里面，包括：

```text
Dataset
DataLoader
Model
Train
Validation
Test
Checkpoint
Plot
```

随着代码增加，文件越来越难阅读。

因此最后进行了第一次项目模块化：

```text
data.py
model.py
engine.py
main.py
```

并把重复的 Validation 和 Test 逻辑统一成：

```python
evaluate()
```

这也是第一次真正体会到函数封装和模块化的作用。

---

## 16. 当前项目最终状态

目前这个 SimpleCNN 项目已经完成：

```text
[x] OrganAMNIST 数据读取
[x] DataLoader
[x] SimpleCNN
[x] Conv / ReLU / MaxPool
[x] Fully Connected Layer
[x] Dropout
[x] CrossEntropyLoss
[x] Adam Optimizer
[x] Batch Training
[x] Epoch Training
[x] Validation
[x] Test
[x] Best Checkpoint
[x] Loss Curve
[x] Accuracy Curve
[x] Precision
[x] Recall
[x] F1-score
[x] Macro-F1
[x] Confusion Matrix
[x] data.py / model.py / engine.py / main.py 模块化
```

到这里，这个项目不再继续增加 CNN 层数或继续针对 Test Accuracy 调参。

它将作为后续实验的：

```text
SimpleCNN Baseline V1
```

保留下来。

---

## 17. 下一步学习计划

接下来准备进入 ResNet18。

学习路线暂定为：

```text
SimpleCNN Baseline
        ↓
为什么普通 CNN 不能简单无限加深？
        ↓
Deep Network 的退化问题
        ↓
Residual Learning
        ↓
Residual Block
        ↓
F(x) + x
        ↓
ResNet18
        ↓
OrganAMNIST + ResNet18
        ↓
与 SimpleCNN Baseline 对比
```

在理解 ResNet18 以后，再继续学习：

```text
CNN Encoder
↓
Feature / Embedding
↓
Few-Shot Learning
↓
N-way K-shot
↓
Support / Query
↓
ProtoNet
↓
Federated Learning
```

目前不急着同时学习很多不同方向。

先把每一个阶段真正写出来、跑起来、理解以后，再进入下一阶段。

---

## 18. 总结

这是我的第一个完整 PyTorch CNN 分类项目。

最终测试结果并不是这个项目唯一重要的部分。

更重要的是通过这个项目第一次完整经历了：

```text
不会写训练循环
↓
理解 Batch
↓
理解 Epoch
↓
自己完成 CNN Training Loop
↓
加入 Validation
↓
加入 Test
↓
发现过拟合
↓
加入 Dropout
↓
保存 Best Checkpoint
↓
画训练曲线
↓
加入 Macro-F1
↓
分析 Confusion Matrix
↓
拆分项目结构
```

过程中也出现过代码写错、Validation/Test 结果异常、对指标理解错误以及不知道如何组织代码等问题。

这些问题没有从项目记录中删除，因为它们本身就是学习过程的一部分。

当前版本作为：

**SimpleCNN Baseline V1**

到这里正式结束。

下一站：**ResNet18**。
