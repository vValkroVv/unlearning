from importlib import import_module
from typing import Any, Dict

from omegaconf import DictConfig


EVALUATOR_REGISTRY: Dict[str, Any] = {}
_EVALUATOR_IMPORTS = {
    "TOFUEvaluator": ("evals.tofu", "TOFUEvaluator"),
    "MUSEEvaluator": ("evals.muse", "MUSEEvaluator"),
    "LMEvalEvaluator": ("evals.lm_eval", "LMEvalEvaluator"),
    "DUETEvaluator": ("evals.duet", "DUETEvaluator"),
}


def _register_trainer(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def _load_evaluator_class(name: str):
    evaluator_class = EVALUATOR_REGISTRY.get(name)
    if evaluator_class is not None:
        return evaluator_class
    module_name, attr_name = _EVALUATOR_IMPORTS[name]
    evaluator_class = getattr(import_module(module_name), attr_name)
    EVALUATOR_REGISTRY[name] = evaluator_class
    return evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    if evaluator_handler_name not in _EVALUATOR_IMPORTS:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    eval_handler = _load_evaluator_class(evaluator_handler_name)
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators
