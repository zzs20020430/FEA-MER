from torchmetrics import ConfusionMatrix, Accuracy, AUROC, F1Score, MeanMetric, Recall, Precision
import torch.nn as nn
import torch
if __name__=="__main__":
    from tools import *
else:
    from utils.tools import*

class DiscriminatorMetrics(nn.Module):
    def __init__(self):
        super(DiscriminatorMetrics,self).__init__()
        # self.train_loss=MeanMetric(task="binary")
        # self.val_loss=MeanMetric(task="binary")
        # self.acc=Accuracy(task="binary")
        # self.auc=AUROC(task="binary")
        # self.confusion_matrix=ConfusionMatrix(task="binary")
        # self.f1=F1Score(task="binary")
        self.metrics=nn.ModuleDict({
            "acc":Accuracy(task="binary"),
            "f1":F1Score(task="binary"),
            "auc":AUROC(task="binary"),
            # "precision":Precision(task="binary"),
            # "recall":Recall(task="binary"),
            "confusion_matrix":ConfusionMatrix(task="binary"),
            "train_loss":MeanMetric(),
            "val_loss":MeanMetric()
        })
        self.train_metrics=["train_loss"]
        # val_metrics=["val_loss","acc","auc","f1","precision","recall","confusion_matrix"]
        self.val_metrics=["val_loss","acc","auc","f1"]
        self.metric_polarity={"val_loss":-1,"train_loss":-1,"acc":1,"auc":1,"f1":1,"check_out":1,"false_alarm":-1}
        self.criterion=nn.CrossEntropyLoss()

    @torch.no_grad()
    def compute_metrics(self):
        metric_list=self.train_metrics if self.training else self.val_metrics
        results=DotDict({})
        for metric in metric_list:
            if metric=="confusion_matrix":
                confusion_matrix=self.metrics[metric].compute()
                tp = confusion_matrix[1][1].item()
                fp = confusion_matrix[0][1].item()
                fn = confusion_matrix[1][0].item()
                tn = confusion_matrix[0][0].item()
                results.check_out = tp / (fp + tp) if (fp + tp) > 0 else 0.0  # 说谎识别率
                results.false_alrm = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # 实话 误判为谎言的概率
            else:
                results[metric] = self.metrics[metric].compute().item()
            self.metrics[metric].reset()

        return results

    @torch.no_grad()
    def forward(self,logits,labels):
        softmax=nn.Softmax(dim=1)
        if self.training:
            self.metrics.train_loss(self.criterion(logits,labels).item())
        else:
            prob=softmax(logits)
            pred=torch.argmax(logits,dim=1)
                        
            # 记录验证损失
            loss = self.criterion(logits, labels)
            # 更新验证指标
            self.metrics['auc'](prob[:,1], labels)
            self.metrics['acc'](pred, labels)
            self.metrics['f1'](pred, labels)
            self.metrics['confusion_matrix'](pred, labels)
            self.metrics['val_loss'](loss.item())


class MeldMetrics(nn.Module):
    def __init__(self):
        super(MeldMetrics,self).__init__()
        self.metrics=nn.ModuleDict({
            "acc":Accuracy(task="multiclass"),
            "f1":F1Score(task="multiclass"),
            "confusion_matrix":ConfusionMatrix(task="multiclass"),
            "train_loss":MeanMetric(),
            "val_loss":MeanMetric()
        })
        self.train_metrics=["train_loss"]
        self.val_metrics=["val_loss","acc","f1"]
        self.metric_polarity={"val_loss":-1,"train_loss":-1,"acc":1,"auc":1,"f1":1}
        self.criterion=nn.CrossEntropyLoss()

    @torch.no_grad()
    def compute_metrics(self):
        metric_list=self.train_metrics if self.training else self.val_metrics
        results=DotDict({})
        for metric in metric_list:
            results[metric] = self.metrics[metric].compute().item()
            if metric == "confusion_matrix":
                results["class_acc"] = self.metrics["confusion_matrix"].compute().diag() / self.metrics["confusion_matrix"].compute().sum(dim=1)
            self.metrics[metric].reset()
        return results

    @torch.no_grad()
    def forward(self,logits,labels):
        if self.training:
            self.metrics.train_loss(self.criterion(logits,labels).item())
        else:
            pred=torch.argmax(logits,dim=1)
            # 记录验证损失
            loss = self.criterion(logits, labels)
            # 更新验证指标
            self.metrics['acc'](pred, labels)
            self.metrics['f1'](pred, labels)
            self.metrics['confusion_matrix'](pred, labels)
            self.metrics['val_loss'](loss.item())
