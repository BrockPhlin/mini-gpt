# mini-gpt

[我观看的课程视频（哔哩哔哩）](https://www.bilibili.com/video/BV1mqrTBvEaf/)

这是我学习神经网络和 GPT 基础过程中的一个小项目。

这是一个学习过程中的项目，代码会随着学习不断修改和完善。

## 项目结构

```text
mini-gpt/
├── autograd/ #自动微分
│   ├── micrograd.py
│   └── using_pytorch.py
├── neural_network/ #MLP感知机实现
│   └── nn.py
├── README.md
└── .gitignore
```

## 如何运行
建议先创建 Python 虚拟环境：
```
python3 -m venv venv
source venv/bin/activate
```
安装项目依赖：
```
pip install torch numpy matplotlib graphviz
```
运行自动微分示例：
```
python3 autograd/micrograd.py
```
运行 MLP 示例：
```
python3 neural_network/nn.py
```
如果使用 Graphviz 绘制计算图，还需要安装 Graphviz 程序本身。
Ubuntu 下可以运行：
```
sudo apt install graphviz
```
运行 nn.py 后，如果代码中调用了：
```
dot.render("graph")
```
通常会在当前目录生成 Graphviz 相关的图文件。