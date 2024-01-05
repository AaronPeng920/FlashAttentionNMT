# FlashAttentionNMT: 基于FlashAttention的神经机器翻译

该仓库是中山大学2023年秋季高级计算机体系结构的课程设计，根据论文 [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) 进行实现和改进。受计算资源的限制，本仓库对原论文中的实验进行简化，设计了该仓库的神经机器翻译任务，两者都是基于 Transformer 的，故在验证 FlashAttention 的设计思想上是等效的。

## 安装

要求：

* 目前仅支持 A100、RTX 3090、RTX 4090、H100 显卡
* 已经安装了 Anaconda 3，如果没有，请前往 [Anaconda](https://www.anaconda.com/download) 进行下载安装
* 需要 CUDA 版本 >= 11.6
* 确保已经根据自己的 CUDA 版本安装了 PyTorch，由于不知道宿主机的 CUDA 版本，无法提供具体的 PyTorch 版本，请在宿主机上输入 `cudnn --version` 查看自己的 CUDA 版本，并根据自己的 CUDA 版本前往 [Torch](https://pytorch.org/) 查询安装命令并安装

### 创建 FlashAttention 虚拟环境

1. 创建虚拟环境

```
conda create -n flash_attn python=3.10
conda activate flash_attn
```

2. 安装 FlashAttention

```
pip install flash-attn==1.0.9
```

3. 安装相关依赖

```
pip install -r requirements.txt
```

### 创建和安装改进的 FlashAttention 虚拟环境

1. 创建虚拟环境

```
conda create -n imp_flash_attn python=3.10
conda activate imp_flash_attn
```

2. 安装改进的 FlashAttention

```
pip install flash-attn
```

3. 安装相关依赖

```
pip install -r requirements.txt
```

## 训练

通过在不同的虚拟环境下训练模型，可以实现不同的 FlashAttention 进行训练：

```shell
# 在 FlashAttention 下训练模型
conda activate flash_attn

# 在改进的 FlashAttention 下训练模型
conda activate imp_flash_attn

# 开始对模型进行训练
python main.py --attn_type `attention type`
```

其中：

`attn_type`: 可以是 `dotscale`, `flash_attn`, `imp_flash_attn`，分别代表原始的点积缩放注意力、FlashAttention 以及改进的 FlashAttention。请注意，由于两个环境的依赖是冲突的，所以请确保在 `flash_attn` 虚拟环境下的 `attn_type` 参数取值是 `flash_attn`，而在 `imp_flash_attn` 虚拟环境下的取值是 `imp_flash_attn`。

训练过程中，在日志 `experiment/train.log` 中会记录每个 epoch 的耗时和损失等信息。

## 测试

在我们的语料库中的序列平均长度不超过 200，在这种配置下难以表现出改进的 FlashAttention 在序列长度维度的并行性，所以我们使用序列长度分别为 `200, 400, 800, 1000, 2000` 的数据进行模拟，相关代码在 `test.py` 中，可以使用下面的命令来进行测试：

```shell
python test.py --attn_type `attention_type` --epochs `epoch num` --seq_len `sequence length`
```

其中：

* `attn_type`: 可以是 `dotscale`, `flash_attn`, `imp_flash_attn`，分别代表原始的点积缩放注意力、FlashAttention 以及改进的 FlashAttention。请注意，由于两个环境的依赖是冲突的，所以请确保在 `flash_attn` 虚拟环境下的 `attn_type` 参数取值是 `flash_attn`，而在 `imp_flash_attn` 虚拟环境下的取值是 `imp_flash_attn`，默认 `dotscale`
* `epochs`: 注意力计算的次数，可以设置为 1000 及以上的数值，默认 1000
* `seq_len`: 序列长度，默认 `-1`，即事先设置的一系列序列长度 `200, 400, 800, 1000, 2000` ，即论文中的数据。可以将此数据设置为其他整数（注意不要太大，可能会导致OOM）









