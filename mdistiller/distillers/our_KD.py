
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def mse_kd_Loss(logits_student, logits_teacher, target, temperature):
    """計算MSE損失"""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    # 提取目標類別的概率
    target_prob_student = pred_student[:, 0]  # 形狀為 [64]
    target_prob_teacher = pred_teacher[:, 0]  # 形狀為 [64]
    # 計算目標類別的交叉熵損失
    mse_loss = F.mse_loss(target_prob_student, target_prob_teacher)

    return mse_loss

def mae_kd_Loss(logits_student, logits_teacher, target, temperature):
    """計算MAE損失"""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    # 提取目標類別的概率
    target_prob_student = pred_student[:, 0]  # 形狀為 [64]
    target_prob_teacher = pred_teacher[:, 0]  # 形狀為 [64]
    # 計算目標類別的交叉熵損失
    mse_loss = F.l1_loss(target_prob_student, target_prob_teacher)

    return mse_loss

def compute_entropy_weights(logits_teacher, temperature):
    """計算entropy reweighting權重"""
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy = -torch.sum(p_t * torch.log(p_t.clamp(min=1e-10)), dim=1)
    return entropy.unsqueeze(1)

def compute_confidence_weight(logits_teacher, temperature, target):
    """計算confidence reweighting權重"""
    gt_mask = _get_gt_mask(logits_teacher, target)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    target_prob = (pred_teacher * gt_mask).sum(dim=1, keepdim=True)
    return target_prob

def compute_dkd_loss(logits_student, logits_teacher, target, temperature):
    """計算DKD損失"""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
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

    return tckd_loss, nckd_loss

def compute_rv_loss(logits_student, logits_teacher, target, temperature):
    """計算Reverse KL損失"""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    # 加入反向 KL
    # 需要計算 teacher * log(teacher/student)
    log_ratio = torch.log(pred_teacher) - torch.log(pred_student)
    reverse_tckd_loss = (pred_teacher * log_ratio).sum(1) * (temperature ** 2)

    return reverse_tckd_loss

def _get_gt_mask(logits, target):
    """獲取目標類別的mask"""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    """獲取非目標類別的mask"""
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    """將mask1和mask2合併"""
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
    """計算soft label的KL散度損失"""
    soft_labels = create_soft_label(target, logits_student.size(1), label_smoothing)
    loss_sl = F.kl_div(
        F.log_softmax(logits_student, dim=1), 
        soft_labels, reduction='batchmean'
    )

    return loss_sl

def contrastive_distillation(logits_student, logits_teacher, temperature):
    """基於概率分布的對比蒸餾"""
    student_probs = F.softmax(logits_student / temperature, dim=1)
    teacher_probs = F.softmax(logits_teacher / temperature, dim=1)
    
    # 計算概率分布的相似度矩陣
    similarity = torch.mm(student_probs, teacher_probs.t())
    
    # 計算對比損失
    labels = torch.arange(logits_student.size(0)).cuda()
    contrastive_loss = F.cross_entropy(similarity, labels)
    
    return contrastive_loss * (temperature ** 2)

def adaptive_temperature(logits_student, logits_teacher, base_temp):
    """根據logits分布差異動態調整溫度"""
    # 計算KL散度
    logits_student = normalize(logits_student)
    logits_teacher = normalize(logits_teacher)
    student_probs = F.softmax(logits_student, dim=1)
    teacher_probs = F.softmax(logits_teacher, dim=1)
    kl_div = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    
    # 根據KL散度調整溫度
    temp = base_temp * (1 + torch.log1p(kl_div))
    return torch.clamp(temp, min=1.0, max=8.0)

def our_loss(logits_student, logits_teacher, target, temperature, cfg):
    
    #dynamic temperature
    if cfg.DT:
        temperature = adaptive_temperature(logits_student, logits_teacher, temperature)
    else:
        temperature = temperature

    # entropy reweighting
    if cfg.ER:
        entropy = compute_entropy_weights(logits_teacher, temperature)
    else:
        entropy = 1.0

    # confidence reweighting
    if cfg.CR:
        confidence_weight = compute_confidence_weight(logits_teacher, temperature, target)
    else:
        confidence_weight = 1.0

    # mse and mae
    if cfg.MSE:
        tckd_mse_loss = mse_kd_Loss(logits_student, logits_teacher, target, temperature)
    else:
        tckd_mse_loss = 0

    if cfg.MAE:
        tckd_mae_loss = mae_kd_Loss(logits_student, logits_teacher, target, temperature)
    else:
        tckd_mae_loss = 0

    # logits normalization
    if cfg.STD:
        logits_student = normalize(logits_student)
        logits_teacher = normalize(logits_teacher)

    # compute rv loss
    if cfg.RV:
        reverse_tckd_loss = compute_rv_loss(logits_student, logits_teacher, target, temperature)
    else:
        reverse_tckd_loss = 0

    # contrastive loss
    if cfg.CT:
        contrastive_loss = contrastive_distillation(logits_student, logits_teacher, temperature)
    else:
        contrastive_loss = 0

    # compute dkd loss
    tckd_loss, nckd_loss = compute_dkd_loss(logits_student, logits_teacher, target, temperature)

    other_loss = 0
    other_loss = cfg.MSE_WEIGHT * tckd_mse_loss + cfg.MAE_WEIGHT * tckd_mae_loss + cfg.RV_WEIGHT * reverse_tckd_loss + cfg.CT_WEIGHT * contrastive_loss

    return ((cfg.ALPHA * tckd_loss + cfg.BETA * nckd_loss + other_loss) * entropy * confidence_weight).mean()

class our_KD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(our_KD, self).__init__(student, teacher)
        self.total_epochs = cfg.SOLVER.EPOCHS
        self.cfg = cfg.OURKD
        self.temperature = [1.0, 2.0, 3.0, 4.0] if self.cfg.MT else [4.0]
        self.mtls_list = [0.1, 0.2] if self.cfg.MTLS else [0.1]

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        # cross-entropy loss
        loss_ce = self.cfg.CE_WEIGHT * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        # knowledge distillation loss for weak augmentation
        loss_dkd_weak_list = []
        for temperature in self.temperature:
            loss = min(kwargs["epoch"] / self.cfg.WARMUP, 1.0) * our_loss(
                logits_student_weak,
                logits_teacher_weak,
                target,
                temperature,
                self.cfg,
            )
            loss_dkd_weak_list.append(loss)
        
        loss_dkd_weak = torch.mean(torch.stack(loss_dkd_weak_list))

        # knowledge distillation loss for strong augmentation
        loss_dkd_strong_list = []
        for temperature in self.temperature:
            loss = min(kwargs["epoch"] / self.cfg.WARMUP, 1.0) * our_loss(
                logits_student_strong,
                logits_teacher_strong,
                target,
                temperature,
                self.cfg,
            )
            loss_dkd_strong_list.append(loss)

        loss_dkd_strong = torch.mean(torch.stack(loss_dkd_strong_list))

        loss_dkd = loss_dkd_weak + loss_dkd_strong

        # label smoothing loss
        if self.cfg.LS:
            # soft label loss for weak augmentation
            loss_ls_weak_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = soft_kl_loss(logits_student_weak, target, ls_ratio)
                loss_ls_weak_list.append(loss_ls)
            
            loss_ls_weak = torch.mean(torch.stack(loss_ls_weak_list))

            # soft label loss for strong augmentation
            loss_ls_strong_list = []
            for ls_ratio in self.mtls_list:
                loss_ls = soft_kl_loss(logits_student_strong, target, ls_ratio)
                loss_ls_strong_list.append(loss_ls)
            
            loss_ls_strong = torch.mean(torch.stack(loss_ls_strong_list))

            loss_ls = self.cfg.LS_WEIGHT * (loss_ls_weak + loss_ls_strong)
        else:
            loss_ls = 0


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd + loss_ls,
        }
        return logits_student_weak, losses_dict
