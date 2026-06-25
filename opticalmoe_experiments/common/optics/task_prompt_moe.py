from .dataset_switching_moe import (
    DatasetSwitchingASGlobalRouterMoEClassifier,
    DatasetSwitchingSharedD2NNClassifier,
)


class SameInputTaskPromptMoEClassifier(DatasetSwitchingASGlobalRouterMoEClassifier):
    """Same-input multitask wrapper around the shared task-prompt MoE."""


class SameInputSharedD2NNClassifier(DatasetSwitchingSharedD2NNClassifier):
    """Same-input multitask wrapper around the shared D2NN multi-head baseline."""
