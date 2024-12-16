
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def our_loss(logits_student, logits_teacher, target, alpha, beta, temperature, er, std, mse, mae, rv):
    
    # entropy reweighting
    if er:
        # get entropy 
        _p_t = F.softmax(logits_teacher / temperature, dim=1)
        entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    if mse:
        prob_student = F.softmax(logits_student / temperature, dim=1)
        prob_teacher = F.softmax(logits_teacher / temperature, dim=1)
        prob_student = cat_mask(prob_student, gt_mask, other_mask)
        prob_teacher = cat_mask(prob_teacher, gt_mask, other_mask)
        # 提取目標類別的概率
        target_prob_student = prob_student[:, 0]  # 形狀為 [64]
        target_prob_teacher = prob_teacher[:, 0]  # 形狀為 [64]
        # 計算目標類別的交叉熵損失
        tckd_ce_loss = F.mse_loss(target_prob_student, target_prob_teacher)

    if mae:
        prob_student = F.softmax(logits_student / temperature, dim=1)
        prob_teacher = F.softmax(logits_teacher / temperature, dim=1)
        prob_student = cat_mask(prob_student, gt_mask, other_mask)
        prob_teacher = cat_mask(prob_teacher, gt_mask, other_mask)
        # 提取目標類別的概率
        target_prob_student = prob_student[:, 0]  # 形狀為 [64]
        target_prob_teacher = prob_teacher[:, 0]  # 形狀為 [64]
        # 計算目標類別的交叉熵損失
        tckd_ce_loss = F.l1_loss(target_prob_student, target_prob_teacher)

    # logits normalization
    if std:
        logits_student = normalize(logits_student)
        logits_teacher = normalize(logits_teacher)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    # tckd
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (temperature ** 2)
    )

    if rv:
        # 加入反向 KL
        # 需要計算 teacher * log(teacher/student)
        log_ratio = torch.log(pred_teacher) - torch.log(pred_student)
        reverse_tckd_loss = (pred_teacher * log_ratio).sum(1) * (temperature ** 2)
        tckd_loss += reverse_tckd_loss

    # nckd
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature * other_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature * other_mask, dim=1
    ) 
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
        * (temperature ** 2)
    )

    if mse:
        # tckd ce
        if er:
            return ((alpha * tckd_loss + beta * nckd_loss + tckd_ce_loss) * entropy.unsqueeze(1)).mean()
        else:
            return (alpha * tckd_loss + beta * nckd_loss + tckd_ce_loss).mean()
    else:
        if er:
            return ((alpha * tckd_loss + beta * nckd_loss) * entropy.unsqueeze(1)).mean()
        else:
            return (alpha * tckd_loss + beta * nckd_loss).mean()

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def create_soft_label(target, num_classes, label_smoothing):
    """
    將one-hot label轉換為soft label
    label_smoothing: 平滑化參數
    """
    one_hot = F.one_hot(target, num_classes).float()
    soft_label = one_hot * (1 - label_smoothing) + label_smoothing / num_classes
    return soft_label

def soft_kl_loss(logits_student, target, label_smoothing):
    soft_labels = create_soft_label(target, logits_student.size(1), label_smoothing)
    loss_sl = F.kl_div(
        F.log_softmax(logits_student, dim=1), 
        soft_labels, reduction='batchmean'
    )

    return loss_sl

class our_KD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(our_KD, self).__init__(student, teacher)
        self.total_epochs = cfg.SOLVER.EPOCHS
        self.ce_loss_weight = cfg.OURKD.CE_WEIGHT
        self.alpha = cfg.OURKD.ALPHA
        self.beta = cfg.OURKD.BETA
        self.mt = cfg.OURKD.MT
        self.temperature = [1.0, 2.0, 3.0, 4.0] if self.mt else [cfg.OURKD.T]
        self.warmup = cfg.OURKD.WARMUP
        self.er = cfg.OURKD.ER
        self.std = cfg.OURKD.STD
        self.ls = cfg.OURKD.LS
        self.mtls = cfg.OURKD.MTLS
        self.mtls_list = [0.1, 0.2] if self.mtls else [0.1]
        self.mse = cfg.OURKD.MSE
        self.mae = cfg.OURKD.MAE
        self.rv = cfg.OURKD.RV

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        # cross-entropy loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student_weak, target)
        loss_ce += self.ce_loss_weight * F.cross_entropy(logits_student_strong, target)

        # knowledge distillation loss for weak augmentation
        loss_dkd_weak_list = []
        for temperature in self.temperature:
            loss = min(kwargs["epoch"] / self.warmup, 1.0) * our_loss(
                logits_student_weak,
                logits_teacher_weak,
                target,
                self.alpha,
                self.beta,
                temperature,
                self.er,
                self.std,
                self.mse,
                self.mae,
                self.rv,
            )
            loss_dkd_weak_list.append(loss)
        
        loss_dkd_weak = torch.mean(torch.stack(loss_dkd_weak_list))

        # knowledge distillation loss for strong augmentation
        loss_dkd_strong_list = []
        for temperature in self.temperature:
            loss = min(kwargs["epoch"] / self.warmup, 1.0) * our_loss(
                logits_student_strong,
                logits_teacher_strong,
                target,
                self.alpha,
                self.beta,
                temperature,
                self.er,
                self.std,
                self.mse,
                self.mae,
                self.rv,
            )
            loss_dkd_strong_list.append(loss)

        loss_dkd_strong = torch.mean(torch.stack(loss_dkd_strong_list))

        loss_dkd = loss_dkd_weak + loss_dkd_strong

        # label smoothing for weak augmentation
        if self.ls:
            loss_ls_weak_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = soft_kl_loss(logits_student_weak, target, ls_ratio)
                loss_ls_weak_list.append(loss_ls)
            
            loss_ls_weak = torch.mean(torch.stack(loss_ls_weak_list))

            loss_ls_strong_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = soft_kl_loss(logits_student_strong, target, ls_ratio)
                loss_ls_strong_list.append(loss_ls)
            
            loss_ls_strong = torch.mean(torch.stack(loss_ls_strong_list))

            loss_ls = loss_ls_weak + loss_ls_strong
            loss_dkd += loss_ls


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student_weak, losses_dict
