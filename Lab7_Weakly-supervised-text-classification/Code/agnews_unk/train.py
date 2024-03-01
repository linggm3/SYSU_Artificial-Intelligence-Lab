import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, num_classes_, vocab_size_, embedding_dim_, hidden_dim_=100, num_layers_=3):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size_, embedding_dim_)
        self.lstm = nn.LSTM(embedding_dim_, hidden_dim_, num_layers_, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim_, num_classes_)
        self.drop = nn.Dropout(p=0.2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        last_hidden_state = output[:, -1, :]
        last_hidden_state = self.drop(last_hidden_state)
        output = self.fc(last_hidden_state)
        output = self.sm(output)
        return output


# 有监督学习（有标签数据）
def supervised_train(model, criterion, optimizer, labeled_loader):
    model.train()
    total_ = 0
    right_ = 0
    for i, (texts, real_labels, known_labels) in enumerate(labeled_loader):
        if i % 2500 == 1:
            print(i-1, right_ / total_)
        optimizer.zero_grad()
        outputs = model(texts.cuda())
        # print(outputs.size(), labels.size())
        # 在已知标签上训练（已知标签不一定是正确的）
        loss = criterion(outputs, known_labels.cuda())
        # print(loss)
        loss.backward()
        optimizer.step()
        total_ += 1
        # 在实际标签上评测性能（实际标签是正确的）
        if torch.argmax(outputs.cpu(), dim=1) == real_labels:
            right_ += 1
    return right_ / total_


# 无监督学习（无标签数据）
def unsupervised_train(model, criterion, optimizer, unlabeled_loader):
    to_remove = []
    model.eval()
    total_ = 0
    right_ = 0
    with torch.no_grad():
        for i, (texts, real_labels, known_labels) in enumerate(unlabeled_loader):
            optimizer.zero_grad()
            outputs = model(texts.cuda())
            # 选择置信度最高的类别，作为类别伪标签
            pseudo_labels = torch.argmax(outputs, dim=1)
            # loss = criterion(outputs, pseudo_labels)
            # 置信度足够高时，将数据条目从 无标签数据集 移动到 有标签数据集
            if outputs[0, int(torch.argmax(outputs, dim=1))].detach().cpu().numpy() > 0.98:
                to_remove.append([outputs[0, int(torch.argmax(outputs, dim=1))].detach().cpu().numpy(),
                                  (texts[0], real_labels[0], pseudo_labels.detach().cpu().numpy()[0])])
                # loss.backward()
                # optimizer.step()
            total_ += 1
            # 在实际标签上评测性能（实际标签是正确的）
            if torch.argmax(outputs.cpu(), dim=1) == real_labels:
                right_ += 1
                # print('right ', outputs[0, int(torch.argmax(outputs, dim=1))])
                # print('wrong ', outputs[0, int(torch.argmax(outputs, dim=1))])
    return to_remove
