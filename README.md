# 代码文档

### 依赖的库：

numpy

pandas

### 运行：

python main.py

数据集请按照：

x1, x2, x3,...., label, index的方式提供

运行输出：

![image](https://github.com/yx0119/SVM/blob/master/image/1574489210464.png)

### 代码说明：

#### 运行设置部分：

![image](https://github.com/yx0119/SVM/blob/master/image/1574487535207.png)

#### run函数：

参数：X数据的特征集，y标签集， train_rate训练集的分割比例，kernel核函数类型（该svm只提供了线性和高斯核），C容忍度，max_iter最大迭代次数，gama（高斯核的参数）

![image](https://github.com/yx0119/SVM/blob/master/image/1574487825692.png)

#### train函数：

初始化：训练集大小，核函数K(i,j), b，w(为了加快线性svm的速度)

![image](https://github.com/yx0119/SVM/blob/master/image/1574487934704.png)

进入循环，优化α

每一轮一次选取所有的αi，判断是否满足KKT条件，满足则不选择该α，continue

![image](https://github.com/yx0119/SVM/blob/master/image/1574488453054.png)

![image](https://github.com/yx0119/SVM/blob/master/image/1574488609967.png)

![image](https://github.com/yx0119/SVM/blob/master/image/1574488701414.png)

KKT条件判断函数

![image](https://github.com/yx0119/SVM/blob/master/image/1574488732665.png)

获取Kij的函数

![image](https://github.com/yx0119/SVM/blob/master/image/1574488754798.png)

获取f(xi)

![image](https://github.com/yx0119/SVM/blob/master/image/1574488793267.png)

分类器

![image](https://github.com/yx0119/SVM/blob/master/image/1574488806847.png)
