import argparse
import copy
import random
from datetime import datetime
import json
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2
import swanlab

from utils import *
from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model

# 替换导入：使用新的层级化TOMCAT函数
from swan_test_hitomcat import predict_logits_text_first_with_hitomcat

cudnn.benchmark = True

class Evaluator:
    """保持原有Evaluator类不变"""
    def __init__(self, dset, model, device):
        self.dset = dset
        self.device = device

        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        if dset.phase == 'train':
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,
            obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        self.score_model = self.score_manifold_model

    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):
        def get_pred_from_scores(_scores, topk):
            _, pair_pred = _scores.topk(
                topk, dim=1)
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )
        scores[~mask] += bias

        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        attr_pred, obj_pred = scores

        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(self.device)

        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        _, pair_pred = closed_scores.topk(topk, dim=1)
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        scores = predictions["scores"]
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        unseen_score_diff = max_seen_scores - correct_scores

        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats

def predict_logits(model, dataset, config):
    """保持原有函数不变"""
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).to(config.device)
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size_wo_tta,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()

    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            data[0] = data[0].to(config.device)
            pairs = pairs.to(config.device)
            with autocast(dtype=torch.bfloat16):
                logits = model(data, pairs)
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            if isinstance(logits, list):
                for i in range(len(logits)):
                    logits[i] = logits[i].cpu()
            else:
                logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt

def predict_logits_text_first(model, dataset, config):
    """保持原有函数不变"""
    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in pairs_dataset]).to(config.device)
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size_wo_tta,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    with torch.no_grad():
        text_feats = []
        num_text_batch = pairs.shape[0] // config.text_encoder_batch_size
        for i_text_batch in range(num_text_batch):
            cur_pair = pairs[i_text_batch * config.text_encoder_batch_size:(
                     i_text_batch + 1) * config.text_encoder_batch_size,
                       :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            text_feats.append(cur_text_feats)
        if pairs.shape[0] % config.text_encoder_batch_size != 0:
            cur_pair = pairs[num_text_batch * config.text_encoder_batch_size:, :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            text_feats.append(cur_text_feats)

        text_feats = torch.cat(text_feats, dim=0)

    for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
    ):
        data[0] = data[0].to(config.device)

        _, clip_logits = model.forward_for_open(data, text_feats)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        clip_logits = clip_logits.cpu()

        all_logits = torch.cat([all_logits, clip_logits], dim=0)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt

def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """保持原有函数不变"""
    logits=logits.detach()
    score = copy.deepcopy(logits)
    mask = (feasiblity >= threshold).float()
    score = score * (mask + seen_mask)

    return score

def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """保持原有函数不变"""
    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=1e3, topk=1
    )

    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=1,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

    return stats

if __name__ == "__main__":
    config = parser.parse_args()
    if config.cfg:
        load_args(config.cfg, config)

    # SwanLab初始化
    if config.use_wandb:  
        swanlab.init(
            project='TTA-' + config.dataset,
            config=vars(config),
            mode=config.swanlab_mode if hasattr(config, 'swanlab_mode') else 'online'
        )
    
    # 设置随机种子
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(config.seed)
    
    print("----")
    test_type = 'OPEN WORLD' if config.open_world else 'CLOSED WORLD'
    print(f"{test_type} evaluation details")
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Beginning Time：" + str(formatted_now))
    print("Programme Path: ", os.path.abspath(sys.argv[0]))
    print("Running Command: ", " ".join(sys.argv))
    print(config)

    dataset_path = config.dataset_path

    print('loading test dataset')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)
    allattrs = test_dataset.attrs
    allobj = test_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).to(config.device)
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model, map_location='cpu'))
    print('loaded model')
    
    # 关键修改：使用新的层级化TOMCAT函数
    if (hasattr(config, 'text_first') and config.text_first):
        print('text first (Hi-TOMCAT)')  # 标识使用层级化版本
        if config.use_tta:
            predict_logits_func = predict_logits_text_first_with_hitomcat  # 替换为新函数
        else:
            predict_logits_func = predict_logits_text_first

    print('evaluating on the test set')
    if config.open_world and config.threshold is None:
        evaluator = Evaluator(test_dataset, model=None, device=config.device)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        seen_mask = test_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        thresholds = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = 0.
        best_th = -10
        test_stats = None
        
        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(
                model, test_dataset, config)
        for th in thresholds:
            temp_logits = threshold_with_feasibility(
                all_logits, test_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
            results = test(
                test_dataset,
                evaluator,
                temp_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )
            auc = results['AUC']
            if auc > best_auc:
                best_auc = auc
                best_th = th
                print('New best AUC', best_auc)
                print('Threshold', best_th)
                test_stats = copy.deepcopy(results)
    else:
        evaluator = Evaluator(test_dataset, model=None,  device = config.device)

        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(
                model, test_dataset, config)
        if config.open_world:
            feasibility_path = os.path.join(
                DIR_PATH, f'data/feasibility_{config.dataset}.pt')
            unseen_scores = torch.load(
                feasibility_path,
                map_location='cpu')['feasibility']
            best_th = config.threshold
            print('using threshold: ', best_th)
            all_logits = threshold_with_feasibility(
                all_logits, test_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
        results = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
        test_stats = copy.deepcopy(results)
        result = ""
        for key in test_stats:
            result = result + key + "  " + str(round(test_stats[key] * 100, 2)) + " | "
        print(result)

    # SwanLab日志
    if config.open_world is False and config.use_wandb:
        swanlab.log(test_stats)

    results = {
        'test': test_stats,
    }

    if config.open_world and best_th is not None:
        results['best_threshold'] = best_th

    if config.load_model:
        title = config.load_model_path
    else:
        os.makedirs(config.save_path, exist_ok=True)
        title = config.save_path + '/'
    if config.open_world:
        result_path = title + "open.calibrated.json"
        print(results)
    else:
        result_path = title + "closed.json"

    with open(result_path, 'w+') as fp:
        json.dump(results, fp)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Ending Time：" + str(formatted_now))
    print("Done!")