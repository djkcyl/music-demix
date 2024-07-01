# Music Source Separation with Band-Split RoPE Transformer (BS-RoFormer)

本项目实现了来自字节跳动 AI 实验室的 Band-Split RoPE Transformer (BS-RoFormer)，这是目前最先进的注意力网络，用于音乐源分离。BS-RoFormer 在 Sound Demixing Challenge (SDX23) 的 MSS 赛道中取得了第一名，并且大幅度超越了之前的第一名。

BS-RoFormer 采用频率域的方法，通过一个频带分割模块将输入的复数频谱投影到子频带级别的表示，然后使用分层变压器堆栈来建模频带内和频带间的序列，以进行多频带掩码估计。

## 模型下载

您可以从以下链接下载预训练模型并放置于项目根目录：[model_bs_roformer_ep_317_sdr_12.9755.ckpt](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt)

## 安装指南

### 使用 Conda 环境安装

1. 克隆项目仓库并进入项目目录：
    ```shell
    git clone https://github.com/djkcyl/music-demix
    cd music-demix
    ```

2. 创建并激活 Conda 环境：
    ```shell
    conda create -n demix python=3.12
    conda activate demix
    ```

3. 安装 PyTorch 及相关依赖：
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. 安装项目依赖：
    ```shell
    pip install pdm
    pdm install
    ```

### 使用 Python 虚拟环境安装

1. 克隆项目仓库并进入项目目录：
    ```shell
    git clone https://github.com/djkcyl/music-demix
    cd music-demix
    ```

2. 创建并激活虚拟环境：
    ```shell
    python -m venv .venv
    .venv\Scripts\Activate.ps1  # Windows
    # source .venv/bin/activate  # macOS/Linux
    ```

3. 安装项目依赖：
    ```shell
    pip install -e .
    ```

## 使用指南

1. 在项目目录中创建 `in` 文件夹：
    ```shell
    mkdir in
    ```

2. 将需要提取人声的音频文件复制到 `in` 文件夹中。

3. 运行主程序：
    ```shell
    python main.py
    ```

4. 提取结果将保存在 `out` 文件夹中。
