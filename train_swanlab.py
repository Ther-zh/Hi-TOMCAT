import argparse
import csv
import os
import pickle
import pprint
import sys

from torch.cuda.amp import autocast
import numpy as np
import torch
import tqdm
# ====================== 1. 替换wandb导入为swanlab ======================
import swanlab
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser
from datetime import datetime
from os.path import join as ospj

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *


def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, train_dataloader, scheduler):
    best_val_AUC = 0
    best_test_AUC = 0

    final_model_state = None
    results = []
    train_losses = []

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(config.device)

    # 跳过已训练的epoch，更新学习率调度器
    for epoch in range(0, config.epoch_start):
        scheduler = step_scheduler(scheduler, config, len(train_dataloader)-1, len(train_dataloader))

    # 从指定epoch开始训练
    for epoch in range(config.epoch_start, config.epochs):
        model.train()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (epoch)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch[0] = batch[0].to(config.device)
            if config.use_mixed_precision:
                with autocast(dtype=torch.bfloat16):
                    loss = model(batch, train_pairs)
            else:
                loss = model(batch, train_pairs)

            # 梯度累积：归一化loss
            loss = loss / config.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            # 梯度累积到指定步数后更新参数
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            # 更新学习率调度器
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss)

            # 更新进度条显示
            progress_bar.set_postfix({"train loss": torch.stack(epoch_train_losses[-50:]).mean().item()})
            progress_bar.update()

        # 计算当前epoch的平均训练损失
        each_train_loss = torch.stack(epoch_train_losses).mean()
        progress_bar.close()
        progress_bar.write(f"epoch {epoch} train loss {each_train_loss.item()}")
        train_losses.append(each_train_loss.item())

        # 保存当前epoch的最新权重
        torch.save(model.state_dict(), os.path.join(config.save_path, f"newest_model.pt"))
        config.current_epoch = epoch

        # 记录当前epoch的结果
        result = {}
        result['train_loss'] = each_train_loss.item()

        # 验证集评估
        print("Epoch " + str(epoch) + " Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        for key, value in val_result.items():
            result['val_' + key] = value

        # 测试集评估
        print("Epoch " + str(epoch) + " Evaluating test dataset:")
        test_result = evaluate(model, test_dataset, config)
        for key, value in test_result.items():
            result['test_' + key] = value

        results.append(result)

        # 保存验证集最优权重
        if config.val_metric == 'best_AUC' and val_result['AUC'] > best_val_AUC:
            best_val_AUC = val_result['AUC']
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))

        # 保存测试集最优权重
        if config.val_metric == 'best_AUC' and test_result['AUC'] > best_test_AUC:
            best_test_AUC = test_result['AUC']
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "test_best.pt"))

        # 记录最后一个epoch的模型状态
        if epoch == config.epochs - 1:
            final_model_state = model.state_dict()

        # ====================== 2. 替换wandb.log为swanlab.log ======================
        if config.use_wandb:
            swanlab.log(result, step=epoch)  # step指定epoch，和训练轮次对齐
        print('')

    # 将所有epoch结果写入csv日志
    for epoch in range(len(results)):
        with open(ospj(config.save_path, 'logs.csv'), 'a') as f:
            w = csv.DictWriter(f, results[epoch].keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(results[epoch])
    
    # 保存最终模型
    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    """模型评估函数：计算验证/测试集的各项指标"""
    model.eval()
    evaluator = test.Evaluator(dataset, model=None, device=config.device)
    with torch.no_grad():  # 推理阶段禁用梯度计算，节省显存
        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test.predict_logits_text_first(
                    model, dataset, config)
    
    # 计算评估指标
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    
    # 整理评估结果
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    return test_saved_results


if __name__ == "__main__":
    # 添加SwanLab续接run的参数（关键：必须在parse_args前添加）
    config = parser.parse_args()
    
    # 加载yml配置文件
    if config.cfg:
        load_args(config.cfg, config)

    # ====================== 3. 替换wandb.init为swanlab.init ======================
    if config.use_wandb:
        # 适配原wandb_net的离线/在线模式：online→online，其他→offline
        swanlab_mode = "online" if config.wandb_net == "online" else "offline"
        # 初始化SwanLab实验（核心修改：添加resume参数+固定name）
        swanlab.init(
            project=f"Troika-{config.dataset}",  # 沿用原wandb的project命名规则
            workspace="zzhTher",  # 你的SwanLab workspace名称
            config=vars(config),  # 传入所有配置参数
            mode=swanlab_mode,    # 离线/在线模式
            name="run-20260119_170627",  # 固定为第一次run的名称（从SwanLab页面复制）
            resume=config.swanlab_resume  # 续接指定ID的run
        )

    # 打印训练基础信息
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Beginning Time：" + str(formatted_now))
    print("Programme Path: ", os.path.abspath(sys.argv[0]))
    print("Running Command: ", " ".join(sys.argv))
    # 打印所有配置参数
    for k, v in vars(config).items():
        print(k, ': ', v)

    # 创建保存目录
    os.makedirs(config.save_path, exist_ok=True)

    # 设置随机种子
    set_seed(config.seed)

    # 加载数据集
    dataset_path = config.dataset_path
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample,
                                       )
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     )
    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural',
                                      )
    
    # 创建训练数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # 提取属性和类别信息
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    # 初始化模型、优化器、学习率调度器
    model = get_model(config, attributes=attributes, classes=classes, offset=offset)
    model.to(config.device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))

    # 加载预训练权重（续训时）
    if config.epoch_start > 0:
    # 修复权重加载的 dtype 错误
        try:
            # 加载权重并指定设备，强制转换为float32兼容
            state_dict = torch.load(config.load_model, map_location=config.device)
            # 遍历权重，修复dtype异常
            for k, v in state_dict.items():
                if v.dtype == torch.bfloat16:
                    state_dict[k] = v.float()
            model.load_state_dict(state_dict)
            print(f"成功加载权重：{config.load_model}")
        except Exception as e:
            print(f"加载权重失败：{e}")
            raise

    try:
        # 开始训练
        train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, train_dataloader, scheduler)

    finally:
        # 保存最终配置
        write_json(os.path.join(config.save_path, "config.json"), vars(config))

        # ====================== 4. 替换wandb.finish为swanlab.finish ======================
        if config.use_wandb:
            swanlab.finish()

        # 打印结束信息
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Ending Time：" + str(formatted_now))
        print("Done!")