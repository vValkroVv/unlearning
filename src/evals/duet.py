from evals.base import Evaluator


class DUETEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        eval_name = eval_cfg.get("name", "DUET")
        super().__init__(eval_name, eval_cfg, **kwargs)
