
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, std, std2):
    if std2:
        logits_student_part2 = normalize(logits_student) if std else logits_student
        logits_teacher_part2 = normalize(logits_teacher) if std else logits_teacher
    else:
        logits_student = normalize(logits_student) if std else logits_student
        logits_teacher = normalize(logits_teacher) if std else logits_teacher

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    if std2:
        pred_teacher_part2 = F.softmax(
            logits_teacher_part2 / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student_part2 / temperature - 1000.0 * gt_mask, dim=1
        )
    else:  
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def er_dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, std, std2):
    # get entropy 
    _p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    # logits normalization
    if std2:
        logits_student_part2 = normalize(logits_student) if std else logits_student
        logits_teacher_part2 = normalize(logits_teacher) if std else logits_teacher
    else:
        logits_student = normalize(logits_student) if std else logits_student
        logits_teacher = normalize(logits_teacher) if std else logits_teacher

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (temperature**2)
    )

    if std2:
        pred_teacher_part2 = F.softmax(
            logits_teacher_part2 / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student_part2 / temperature - 1000.0 * gt_mask, dim=1
        )
    else:  
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
        * (temperature**2)
    )
    return ((alpha * tckd_loss + beta * nckd_loss) * entropy.unsqueeze(1)).mean()

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


class our_KD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(our_KD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.OURKD.CE_WEIGHT
        self.alpha = cfg.OURKD.ALPHA
        self.beta = cfg.OURKD.BETA
        self.temperature = cfg.OURKD.T
        self.warmup = cfg.OURKD.WARMUP
        self.er = cfg.OURKD.ER
        self.std = cfg.OURKD.STD
        self.std2 = cfg.OURKD.STD2

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        if self.er:
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * er_dkd_loss(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.std,
                self.std2
            )
        else:
            # losses
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.std,
                self.std2
            )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
