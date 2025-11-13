from enum import Enum


class PointBackboneType(Enum):
    SCENESCRIPT = "scenescript"
    SONATA = "sonata"


class ProjectorType(Enum):
    LINEAR = "linear"
    MLP = "mlp"
    PATCH_MERGER = "patch_merger"
