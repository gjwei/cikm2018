## 运行本项目的过程：
1. 安装requirements.txt中的所有库
2. 从链接https://stanfordnlp.github.io/CoreNLP/中下载corenlp的安装包（运行需要Java），下载[西班牙语](http://nlp.stanford.edu/software/stanford-spanish-corenlp-2018-02-27-models.jar)的对应文件,放置到安装包主目录中
3. 修改code/build_data.py中的`corenlp_path`

## 运行环境
Ubuntu16.04 

python3

TitanXP

## 文件夹说明
./activations/  激活函数

./checkponts/  保留的权值文件

./data/   模型加载数据的代码

./fasttext/ 词向量位置

./ml_method/  尝试使用machine learning方法提取特征加到深度模型中，但是提取到的特征加到模型后模型很难收敛。最后没有用到

./result/  保存结果

./script/  放置废弃不用的代码

./utils/  工具代码


### 运行
1. 打开code/
2. 运行 `python build_data.py`,数据预处理
3. 运行 `generate_embedding_weights.py`, 根据vocab生成预训练好的词向量
2. 修改config.py中的参数，然后运行main-5fold.py，会进行5 fold的CV训练。
3. 通过修改main-5fold.py中的model_class来选择对应的model。
4. 训练多次，运行blending_result.py，将所有结果进行blending
5. 提交结果

