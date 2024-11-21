from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, AugTrainer, AugDOTTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "aug": AugTrainer,
    "aug_dot": AugDOTTrainer
}
