import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from model import *

class dataSet(Dataset):
    def __init__(self, feature_file):
        super(dataSet, self).__init__()
        with open(feature_file, 'rb') as f:
            feature_list = pickle.load(f, encoding='bytes')
            f.close()
        self.feature = feature_list

    def __getitem__(self, index):
        p1, p2, label = self.feature[index]
        return p1, p2, label

    def __len__(self):
        return len(self.feature)

dataset = dataSet(feature_file=featurefile)


def train_epoch(model, train_loader, loss_fn, optimizer):
    train_loss = 0
    true = []
    pred = []
    y_score = []
    model.train()
    for i, data in enumerate(train_loader):
        X1, X2, label = data
        X1 = torch.as_tensor(X1, dtype=torch.float32, device=device)
        X2 = torch.as_tensor(X2, dtype=torch.float32, device=device)
        label = torch.as_tensor(label, dtype=torch.float32, device=device)
        y_pred = model.forward(X1, X2).reshape(-1)
        y_pred = y_pred.to(torch.float32)
        y_score.append(y_pred)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recorded loss
        train_loss += loss.item()

        # calculate acc
        predict = torch.where(y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        true.append(label.item())
        pred.append(predict.item())

    acc = accuracy_score(true, pred)
    recall_scores = recall_score(true, pred)
    f1_scores = f1_score(true, pred)
    precision_scores = precision_score(true, pred)
    mcc = matthews_corrcoef(true, pred)
    auc = roc_auc_score(torch.tensor(true,device='cpu'), torch.tensor(y_score,device='cpu'))
    return acc, train_loss, recall_scores, f1_scores, precision_scores, auc, mcc


def test_epoch(model, test_loader, loss_fn):
    true = []
    pred = []
    y_score = []
    test_loss = 0
    model.eval()
    for i, data in enumerate(test_loader):
        X1, X2, label = data
        X1 = torch.as_tensor(X1, dtype=torch.float32, device=device)
        X2 = torch.as_tensor(X2, dtype=torch.float32, device=device)
        label = torch.as_tensor(label, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred = model.forward(X1, X2).reshape(-1)
        y_pred = y_pred.to(torch.float32)
        y_score.append(y_pred)
        loss = loss_fn(y_pred, label)

        test_loss += loss.item()
        predict = torch.where(y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        true.append(label.item())
        pred.append(predict.item())

    acc = accuracy_score(true, pred)
    recall_scores = recall_score(true, pred)
    f1_scores = f1_score(true, pred)
    precision_scores = precision_score(true, pred)
    mcc = matthews_corrcoef(true, pred)
    auc = roc_auc_score(torch.tensor(true,device='cpu'), torch.tensor(y_score,device='cpu'))

    return acc, test_loss, recall_scores, f1_scores, precision_scores, auc, mcc

device = torch.device("cuda:0")
nums_epoches = 200
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=42)
foldperf = {}
for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
    model = RSPPI()
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
                'train_recall_scores': [], 'test_recall_scores': [], 'train_f1_scores': [], 'test_f1_scores': [],
              'train_precision_scores': [], 'test_precision_scores':[],
              'train_auc':[], 'train_mcc':[],
              'test_auc':[], 'test_mcc':[]}

    for epoch in range(nums_epoches):
        train_acc, train_loss, train_recall_scores, train_f1_scores, train_precision_scores, train_auc, train_mcc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc, test_loss, test_recall_scores, test_f1_scores, test_precision_scores, test_auc, test_mcc = test_epoch(model, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_acc * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_acc * 100
        scheduler.step()
        print(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %Training Recall {:.2f} % Training F1 score {:.2f} % Test Recall {:.2f} % Test F1 score {:.2f} %".format(
                epoch + 1,
                nums_epoches,
                train_loss,
                test_loss,
                train_acc,
                test_acc,
                train_recall_scores,
                train_f1_scores,
                test_recall_scores,
                test_f1_scores))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_recall_scores'].append(train_recall_scores)
        history['train_f1_scores'].append(train_f1_scores)
        history['train_precision_scores'].append(train_precision_scores)
        history['test_recall_scores'].append(test_recall_scores)
        history['test_f1_scores'].append(test_f1_scores)
        history['test_precision_scores'].append(test_precision_scores)
        history['train_auc'].append(train_auc)
        history['train_mcc'].append(train_mcc)
        history['test_auc'].append(test_auc)
        history['test_mcc'].append(test_mcc)
    foldperf['fold{}'.format(fold + 1)] = history