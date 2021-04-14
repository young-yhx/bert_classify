# bert_classify pytorch版
bert新闻十分类模型

## 版本问题
python 3.6  
pytorch 1.7  
transformers 4.3.3  
一块GPU，2次EPOCH训练时间不超30min  

## 数据预处理
数据分train，dev，test，三个txt文件  
转换数据格式  
设置分批训练  
打乱数据顺序    

## 定义模型参数
batch_size 64  
max_len 30  
device cuda  
epochs 2 (自己可修改多轮训练尝试，2轮训练正确率结果94.9%)

### 预训练模型下载 放在bert_model  包含config.json,pytorch_model.bin,vocab_txt
git clone https://huggingface.co/bert-base-chinese  

## 后面附带bert_BiLSTM融合模型代码
主代码为share.ipynb的jupyter文件  
有兴趣的可以看一下bert_BiLSTM的融合代码

## 运行
直接jupyter运行share.ipynb的jupyter类型代码  
或者将share转化成.py文件直接运行

## 参考
https://zhuanlan.zhihu.com/p/112655246  
https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch  

