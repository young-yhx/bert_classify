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
打乱数据书讯

## 定义模型参数
batch_size 64
max_len 30
device cuda
epochs 2 (自己可修改多轮训练尝试，2轮训练正确率结果94.9%)

## 后面附带bert_BiLSTM融合模型代码

class Bert_Model_LSTM(nn.Module):
    def __init__(self,bert_path,classes=10):
        super(Bert_Model_LSTM,self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(768, 768, 2,bidirectional=True, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.fc_rnn = nn.Linear(768*2,classes)  #直接分类
    def forward(self,input_ids,attention_mask=None,token_type_ids=None):
        outputs = self.bert(input_ids,attention_mask,token_type_ids)
        out, _ = self.lstm(outputs[0])
        out = self.dropout(out)
        logit = self.fc_rnn(out[:,-1,:])
        return logit
        
## 参考
