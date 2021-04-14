#模型融合代码
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
