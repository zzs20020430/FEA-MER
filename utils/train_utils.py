from torchmetrics import ConfusionMatrix, Accuracy, AUROC, F1Score, MeanMetric, Recall, Precision
from .tools import DotDict, read_yaml_to_dict, Path, move_to_device, save_dict_to_json, read_json_to_dict, get_datetime
import numpy as np
import os
import torch
import torch.nn as nn
from collections import defaultdict

#torch metrics
train_metrics=["train_loss"]
# val_metrics=["val_loss","acc","auc","f1","precision","recall","confusion_matrix"]
val_metrics=["val_loss","acc","auc","f1"]
metric_polarity={"val_loss":-1,"train_loss":-1,"acc":1,"auc":1,"f1":1,"check_out":1,"false_alarm":-1}   #1:positive, -1:negative
metrics=DotDict({
    "acc":Accuracy(task="binary"),
    "f1":F1Score(task="binary"),
    "auc":AUROC(task="binary"),
    # "precision":Precision(task="binary"),
    # "recall":Recall(task="binary"),
    "confusion_matrix":ConfusionMatrix(task="binary"),
    "train_loss":MeanMetric(task="binary"),
    "val_loss":MeanMetric(task="binary")
})

#arg parser
def parse_args(config_dir:Path, test=False):
    args=DotDict({})
    args.update(read_yaml_to_dict(config_dir/"model/moe.yaml"))                 #    moe_config
    args.update(read_yaml_to_dict(config_dir/"model/train.yaml"))               #    trainer_config
    data_cfg=read_yaml_to_dict(config_dir/"model/data.yaml")                    #    data_config(data_dim, class_num)
    args.update(data_cfg)
    args.num_tasks=len(args.domains)
    args.datasets = []
    args.sampled=[]                                                             #sampled domain
    # load dataset args
    for domain in args.domains:                                                 #moe_config.domains
        dataset_args=read_yaml_to_dict(config_dir/f"dataset/{domain}.yaml")
        dataset_args.batch_size=dataset_args.get('batch_size', args.batch_size)
        dataset_args.update(data_cfg)
        args.datasets.append(dataset_args)
        if dataset_args.get('sampling',False):
            args.sampled.append(domain)
    if test:
        args.update(read_yaml_to_dict(config_dir/"model/test.yaml"))
    return args

#metric evaluation
def compute_metrics(metrics:DotDict, metric_list:list):
    results=DotDict({})
    for metric in metric_list:
        if metric=="confusion_matrix":
            confusion_matrix=metrics[metric].compute()
            tp = confusion_matrix[1][1].item()
            fp = confusion_matrix[0][1].item()
            fn = confusion_matrix[1][0].item()
            tn = confusion_matrix[0][0].item()
            results.check_out = tp / (fp + tp) if (fp + tp) > 0 else 0.0  # 说谎识别率
            results.false_alrm = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # 实话 误判为谎言的概率
        else:
            results[metric] = metrics[metric].compute().item()
        metrics[metric].reset()

    return results

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

#余弦退火学习率: https://blog.csdn.net/weixin_42392454/article/details/127766771
def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def get_trainable_params(args, model, frozen_list_filename=""):
    # get params
    if args.train_mode == "adapter":    # only train adapter,  # default="whole" but choosed adapter
        print("[Training mode] Moe-Adapters")
        for k, v in model.named_parameters():  # forzen params，冻结非adapt参数
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

        if args.get("frozen",False) == True:  # frozen-strategy: frozen some params except adapter params
            print('-------frozen--------')
            with open(args.frozen_dir/frozen_list_filename, "r") as file:
                lines = file.read().splitlines()
                frozen_list = list(set(lines))
            params = []
            params_name = []
            for k, v in model.named_parameters():   #冻结上一次训练的adapt参数 并且requires_grad = False
                if k in frozen_list:
                    v.requires_grad = False
                    continue
                if "adaptmlp" in k or "router" in k or "noise" in k:
                    params.append(v)
                    params_name.append(k)

            # print('frozen mode========trainable params============', params_name)
            # print('frozen mode========frozen params of trainable params============', frozen_list)
        else:                                       # train all adapters                     
            params = [
                v for k, v in model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
            ]
            params_name = [
                k for k, v in model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
            ]
    else:                               # train whole model
        params = [p for p in model.parameters() if p.requires_grad]
        params_name = [k for k, v in model.named_parameters() if v.requires_grad]
    # print trainable params's information
    print('========trainable params============', params_name)
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print('The number of Total Trainable Parameters------------------:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")
    return params
            
def find_expert_modules(module:nn.Module, prefix=""):
    expert_modules = []
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if hasattr(child, "choose_map"):
            expert_modules.append((child, child_prefix))
        expert_modules.extend(find_expert_modules(child, child_prefix))
    return expert_modules

def save_frozen_list(model, frozen_path, expert_modules=None, topk=2):
    if expert_modules is None:
        expert_modules = find_expert_modules(model)
    with open(frozen_path, "a") as file:
        for module, prefix in expert_modules:
            choose_map = module.choose_map
            top_values, top_indices = torch.topk(choose_map, topk)
            for k in range(len(top_indices)):
                item1 = f'{prefix}.adaptmlp_list.{top_indices[k]}.down_proj.weight'
                item2 = f'{prefix}.adaptmlp_list.{top_indices[k]}.down_proj.bias'
                item3 = f'{prefix}.adaptmlp_list.{top_indices[k]}.up_proj.weight'
                item4 = f'{prefix}.adaptmlp_list.{top_indices[k]}.up_proj.bias'
                file.write(item1 + "\n")
                file.write(item2 + "\n")
                file.write(item3 + "\n")
                file.write(item4 + "\n")
    
    print('=======================bingo!=============================')

def init_dirs(args, current_root:Path):
    cur_datetime=get_datetime()
    dir_name=f"{'->'.join(args.domains[task_id] for task_id in args.domain_order)} num_experts_{args.num_experts} topk_{args.topk}{' sampled['+' '.join(args.sampled)+']' if len(args.sampled)>0 else ''} {cur_datetime}"
    
    #先初始化保存路径前缀,拼接相应的filename后使用
    args.ckpt_dir = current_root/f"ckpt/{dir_name}"            #ckpt
    Path.mkdir(args.ckpt_dir,parents=True,exist_ok=True)
    args.frozen_dir =  current_root/f"frozen/{dir_name}"       #frozen_params list
    Path.mkdir(args.frozen_dir,parents=True,exist_ok=True)
    args.log_dir = current_root/f"log/{dir_name}"              #log
    Path.mkdir(args.log_dir,parents=True,exist_ok=True)
    
def get_dirs(args, current_root:Path):                          #获取已经保存工作记录的 工作目录名
    dir_name=args.work_dir
    
    #先初始化保存路径前缀,拼接相应的filename后使用
    args.ckpt_dir = current_root/f"ckpt/{dir_name}"            #ckpt
    args.frozen_dir =  current_root/f"frozen/{dir_name}"       #frozen_params list
    args.log_dir = current_root/f"log/{dir_name}"              #log
def del_empty_dirs(args):
    for dir in [args.ckpt_dir,args.frozen_dir,args.log_dir]:    
        if len(list(dir.iterdir()))==0:
            dir.rmdir()
            
def compute_forget_rate(last_domain_metrics, cur_domain_metrics):
    forget_rate = {}
    metric_set=set(last_domain_metrics.keys()).intersection(cur_domain_metrics.keys())
    for key in metric_set:
        last_value=last_domain_metrics[key]
        if key in cur_domain_metrics:
            cur_value = cur_domain_metrics[key]
            if last_value != 0:
                forget_rate[key] = (last_value - cur_value) / last_value
            else:
                forget_rate[key] = 0.0  # Avoid division by zero
        else:
            # If the metric is not present in the current domain, consider it as 0
            forget_rate[key] = 1.0 if last_value != 0 else 0.0
    
    return forget_rate