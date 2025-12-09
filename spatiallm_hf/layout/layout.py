from typing import List, Dict, Any
import numpy as np
from spatiallm_hf.layout.entity import Room


class Layout:
    def __init__(self, rooms_data: List[Dict[str, Any]]):
        self.rooms = [Room(r) for r in rooms_data]

    def translate(self, translation: np.ndarray):
        """
        Translate all entities by the given vector [x, y, z].
        """
        for room in self.rooms:
            room.translate(translation)

    def rotate(self, angle: float):
        """
        Rotate all entities by the given angle (radians) around the Z axis.
        """
        for room in self.rooms:
            room.rotate(angle)

    def scale(self, scale: float):
        """
        Scale all entities by the given factor.
        """
        for room in self.rooms:
            room.scale(scale)

    def normalize_and_discretize(self, num_bins: int):
        """
        Normalize coordinates to [0, num_bins-1] based on global world bounds.
        """
        for room in self.rooms:
            room.normalize_and_discretize(num_bins)

    def filter_empty_bboxes(self, points: np.ndarray, num_points: int = 100):
        """
        Filter out bounding boxes that contain fewer than `num_points` points.
        """
        # TODO: Implement actual filtering logic if needed.
        # For now, we assume this is a no-op or just keeps all boxes
        # as implementing point-in-box check requires more complex logic.
        pass

    def reorder_entities(self):
        """
        Reorder entities based on some criteria (e.g., spatial position).
        """
        # TODO: Implement reordering logic.
        pass

    def to_token_string(self) -> str:
        """
        Convert the layout to a string representation for tokenization.
        """
        # TODO: Implement actual serialization format.
        import json

        return json.dumps(self.to_list())

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert the layout back to a list of dictionaries.
        """
        return [room.to_dict() for room in self.rooms]
