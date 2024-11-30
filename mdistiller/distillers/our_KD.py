
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def our_loss(logits_student, logits_teacher, target, alpha, beta, temperature, er, std, dt, ce):
    
    # entropy reweighting
    if er:
        # get entropy 
        _p_t = F.softmax(logits_teacher / temperature, dim=1)
        entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    # dynamic temperature
    if dt:
        # DTKD Loss
        epsilon = 1e-8  # 防止除以零
        logits_student_max = logits_student.max(dim=1, keepdim=True)[0]
        logits_teacher_max = logits_teacher.max(dim=1, keepdim=True)[0]
        # 計算溫度比例
        ratio_teacher = (2 * logits_teacher_max) / (logits_teacher_max + logits_student_max + epsilon)
        ratio_student = (2 * logits_student_max) / (logits_teacher_max + logits_student_max + epsilon)
        # 使用 sigmoid 函數限制溫度範圍，避免梯度爆炸
        logits_teacher_temp = ratio_teacher * temperature  # 教師溫度
        logits_student_temp = ratio_student * temperature  # 學生溫度
    else:
        logits_student_temp = temperature
        logits_teacher_temp = temperature

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    if ce:
        prob_student = F.softmax(logits_student / temperature, dim=1)
        prob_teacher = F.softmax(logits_teacher / temperature, dim=1)
        prob_student = cat_mask(prob_student, gt_mask, other_mask)
        prob_teacher = cat_mask(prob_teacher, gt_mask, other_mask)
        # 提取目標類別的概率
        target_prob_student = prob_student[:, 0]  # 形狀為 [64]
        target_prob_teacher = prob_teacher[:, 0]  # 形狀為 [64]
        # 計算目標類別的交叉熵損失
        tckd_ce_loss = F.mse_loss(target_prob_student, target_prob_teacher)

    # logits normalization
    if std:
        logits_student = normalize(logits_student)
        logits_teacher = normalize(logits_teacher)

    pred_student = F.softmax(logits_student / logits_student_temp, dim=1)
    pred_teacher = F.softmax(logits_teacher / logits_teacher_temp, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    # tckd
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (logits_teacher_temp * logits_student_temp)
    )

    # nckd
    pred_teacher_part2 = F.softmax(
        logits_teacher / logits_teacher_temp * other_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / logits_student_temp * other_mask, dim=1
    ) 
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
        * (logits_teacher_temp * logits_student_temp)
    )

    if ce:
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

def create_dynamic_soft_label(target, num_classes, base_smoothing, epoch, total_epochs):
    """
    動態生成軟標籤
    Args:
        target: 原始標籤 [batch_size]
        num_classes: 類別數量
        base_smoothing: 基礎平滑參數
        epoch: 當前訓練回合
        total_epochs: 總訓練回合數
    Returns:
        soft_labels: 動態軟標籤 [batch_size, num_classes]
    """
    # 根據訓練進度計算動態平滑參數
    progress = min(epoch / total_epochs, 1.0)
    dynamic_smoothing = base_smoothing * (1 - progress)
    
    # 生成 one-hot 編碼
    one_hot = F.one_hot(target, num_classes).float()
    
    # 計算軟標籤
    soft_label = one_hot * (1 - dynamic_smoothing) + dynamic_smoothing / num_classes
    
    return soft_label

def dynamic_soft_loss(logits_student, target, base_smoothing, epoch, total_epochs):
    """
    計算動態軟標籤的 KL 散度損失
    Args:
        logits_student: 學生模型的輸出 [batch_size, num_classes]
        target: 原始標籤 [batch_size]
        base_smoothing: 基礎平滑參數
        epoch: 當前訓練回合
        total_epochs: 總訓練回合數
    Returns:
        loss: 動態軟標籤損失
    """
    # 獲取類別數量
    num_classes = logits_student.size(1)
    
    # 生成動態軟標籤
    soft_labels = create_dynamic_soft_label(
        target, num_classes, base_smoothing, epoch, total_epochs
    )
    
    # 計算 KL 散度損失
    log_prob = F.log_softmax(logits_student, dim=1)
    loss = F.kl_div(
        log_prob,
        soft_labels.to(logits_student.device),
        reduction='batchmean'
    )
    
    return loss


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
        self.dls = cfg.OURKD.DLS
        self.mtls = cfg.OURKD.MTLS
        self.mtls_list = [0.1, 0.2] if self.mtls else [0.1]
        self.dt = cfg.OURKD.DT
        self.ce = cfg.OURKD.CE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # cross-entropy loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # knowledge distillation loss
        loss_dkd_list = []
        for temperature in self.temperature:
            loss = min(kwargs["epoch"] / self.warmup, 1.0) * our_loss(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                temperature,
                self.er,
                self.std,
                self.dt,
                self.ce,
            )
            loss_dkd_list.append(loss)
        
        loss_dkd = torch.mean(torch.stack(loss_dkd_list))

        # label smoothing
        if self.ls:
            loss_ls_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = soft_kl_loss(logits_student, target, ls_ratio)
                loss_ls_list.append(loss_ls)
            
            loss_dkd += torch.mean(torch.stack(loss_ls_list))
        elif self.dls:
            loss_ls_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = dynamic_soft_loss(logits_student, target, ls_ratio, kwargs["epoch"], self.total_epochs)
                loss_ls_list.append(loss_ls)
            
            loss_dkd += torch.mean(torch.stack(loss_ls_list))


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
