# data input -> layer(calculate  + activate) -> calculate loss -> optimizer -> output
# Xavier and He/Kaiming Initialization
The goal of both Xavier and He/Kaiming is to ensure that the signal (activations) maintains a consistent "strength" (standard deviation/variance) from the first layer to the last.
特性	Xavier 初始化	He (Kaiming) 初始化
提出者	Xavier Glorot (2010)	何凯明 (2015)
最佳拍档	Tanh, Sigmoid	ReLU, Leaky ReLU
主要逻辑	保持输入输出方差一致	补偿 ReLU 造成的信号损失
方差公式	






