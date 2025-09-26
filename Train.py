import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np
import torchmetrics
import torch.optim as optim
from model import convATTnet
from logger import Logger
import time
from sklearn.model_selection import KFold

logger = Logger()


def log(str, log_out):
    print(str)
    logger.set_filename(log_out)
    logger.log(str + '\n')


if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Dictionary of Amino Acids and Numbers.
codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20"}


def format(predict_fasta, is_train=True):  # 新增is_train参数，区分训练/测试
    formatfasta = []
    recordlabel = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        seq = str(record.seq)
        length = len(seq)
        if length <= 1000:
            # 训练时添加随机padding（而非固定补0），增强鲁棒性
            if is_train:
                pad_length = 1000 - length
                # 随机在序列前后补0（而非固定在前面）
                pad_front = np.random.randint(0, pad_length + 1)
                pad_back = pad_length - pad_front
                fastalist.extend([0] * pad_front)
            else:
                # 测试时保持原有固定padding（确保一致性）
                fastalist.extend([0] * (1000 - length))

            # 训练时随机替换1%的氨基酸（数据增强）
            if is_train and np.random.random() < 0.5:  # 50%概率触发
                seq_list = list(seq)
                for i in range(int(len(seq) * 0.01)):
                    pos = np.random.randint(0, len(seq))
                    # 随机替换为其他氨基酸（非自身）
                    new_aa = np.random.choice([k for k in codeadict.keys() if k != seq_list[pos]])
                    seq_list[pos] = new_aa
                seq = ''.join(seq_list)

            for a in seq:
                fastalist.append(int(codeadict[a]))

            # 训练时补全剩余padding（若使用随机前补）
            if is_train:
                fastalist.extend([0] * pad_back)

        formatfasta.append(fastalist)
        recordlabel.append(1 if record.id.startswith('allergen') else 0)
    inputarray = np.array(formatfasta)
    labelarray = np.array(recordlabel)
    return (inputarray, labelarray)


def validation(model, x_valid, y_valid_label, criterion, args):
    valid_ids = TensorDataset(x_valid, y_valid_label)
    valid_loader = DataLoader(
        dataset=valid_ids, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model.eval()
    accuracy = torchmetrics.Accuracy().to(device)
    recall = torchmetrics.Recall().to(device)  # 默认binary
    precision = torchmetrics.Precision().to(device)
    auroc = torchmetrics.AUROC(num_classes=1).to(device)  # 二分类num_classes=1
    f1 = torchmetrics.F1Score().to(device)

    finaloutputs = torch.tensor([]).to(device)
    finallabels = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        valid_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            labels = torch.as_tensor(labels, dtype=torch.long)
            finaloutputs = torch.cat([finaloutputs, outputs], 0)
            finallabels = torch.cat([finallabels, labels], 0)
        accuracy(finaloutputs, finallabels)
        recall(finaloutputs, finallabels)
        precision(finaloutputs, finallabels)
        auroc(finaloutputs, finallabels)
        f1(finaloutputs, finallabels)
        accuracy_value = accuracy.compute()
        recall_value = recall.compute()
        precision_value = precision.compute()
        auroc_value = auroc.compute()
        f1_value = f1.compute()
        accuracy.reset()
        recall.reset()
        precision.reset()
        auroc.reset()
        f1.reset()
    return (valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重（可根据数据比例调整）
        self.gamma = gamma  # 难分样本聚焦参数

    def forward(self, outputs, labels):
        BCE_loss = F.binary_cross_entropy(outputs, labels.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测概率
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

def train(args):
    x, y = format(args.inputs, is_train=True)  # 修改此处
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    valid_loss_sum, accuracy_sum, recall_sum, precision_sum, auroc_sum, f1_sum = 0, 0, 0, 0, 0, 0
    k = 0
    skf = KFold(n_splits=10, shuffle=True, random_state=42)  # 开启shuffle，使各折数据分布更均匀
    for fold_idx,(train_index, valid_index) in enumerate(skf.split(x)):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train_label, y_valid_label = y[train_index], y[valid_index]
        k = k+1
        model = convATTnet()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 新增L2正则化（weight_decay）
        # 替换criterion为FocalLoss
        criterion = FocalLoss(alpha=0.4, gamma=2).to(device)  # alpha偏向少数类（根据数据调整）

        # 新增学习率调度器：当验证F1连续3轮不提升，学习率乘以0.5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        train_ids = TensorDataset(x_train, y_train_label)
        train_loader = DataLoader(
            dataset=train_ids, batch_size=args.batch_size, shuffle=True, drop_last=True)

        model.train()
        best_f1 = 0
        early_stop_counter = 0  # 早停计数器
        for epoch in range(args.epochs):
            train_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                log('[k: %d, batch: %d] train_loss: %.3f' %
                    (k, i + 1, loss.item()), 'train.log')
                now = time.asctime(time.localtime(time.time()))
            # 验证并获取指标
            valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value = validation(
                model, x_valid, y_valid_label, criterion, args)

            # 学习率调度器更新（基于F1）
            scheduler.step(f1_value)

            # 早停机制：连续5轮F1不提升则停止训练
            if f1_value > best_f1:
                best_f1 = f1_value
                torch.save(model.state_dict(), './model.' + str(k) + '.pt')
                early_stop_counter = 0  # 重置计数器
            else:
                early_stop_counter += 1
                if early_stop_counter >= 5:
                    log(f"Early stopping at epoch {epoch} for fold {k}", 'train.log')
                    break  # 停止当前折的训练
            # Generate validation results for each epoch.
            log('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (epoch, valid_loss, accuracy_value, recall_value,
                precision_value, auroc_value, f1_value), './model.'+str(k)+'.fold.everyepoch.valid.txt')
            # Save the model with the largest F1 value.

        # Generate validation results for each fold
        log('[k: %d] valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
            k, valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value), 'valid.log')
        valid_loss_sum += valid_loss
        accuracy_sum += accuracy_value.item()
        recall_sum += recall_value.item()
        precision_sum += precision_value.item()
        auroc_sum += auroc_value.item()
        f1_sum += f1_value.item()
    log('valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
        valid_loss_sum/10, accuracy_sum/10, recall_sum/10, precision_sum/10, auroc_sum/10, f1_sum/10), 'valid.log')
    now = time.asctime(time.localtime(time.time()))
    log(str(now), 'train.log')
