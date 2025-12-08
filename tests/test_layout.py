import unittest
import numpy as np
from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import Wall, Door, Window, Room, Bbox


class TestLayout(unittest.TestCase):
    def test_layout_without_room(self):
        layout_str = """
wall_0=(0, 0, 0, 10, 0, 0, 3, 0.2)
wall_1=(10, 0, 0, 10, 10, 0, 3, 0.2)
door_0=(wall_0, 5, 0, 1, 1, 2)
window_0=(wall_1, 10, 5, 1.5, 1, 1)
bbox_0=(Chair, 2, 2, 0, 0, 1, 1, 1)
"""
        layout = Layout(layout_str)

        self.assertEqual(len(layout.walls), 2)
        self.assertEqual(len(layout.doors), 1)
        self.assertEqual(len(layout.windows), 1)
        self.assertEqual(len(layout.bboxes), 1)
        self.assertEqual(len(layout.rooms), 0)

        # Check Wall_0
        self.assertEqual(layout.walls[0].id, 0)
        self.assertAlmostEqual(layout.walls[0].bx, 10)

        # Check Door_0
        self.assertEqual(layout.doors[0].id, 0)
        self.assertEqual(layout.doors[0].wall_id, 0)

        # Check Window_0
        self.assertEqual(layout.windows[0].id, 0)
        self.assertEqual(layout.windows[0].wall_id, 1)

    def test_layout_with_room(self):
        layout_str = """
wall_0=(0, 0, 0, 10, 0, 0, 3, 0.2)
wall_1=(10, 0, 0, 10, 10, 0, 3, 0.2)
room_0=(["wall_0", "wall_1"], LivingRoom)
"""
        layout = Layout(layout_str)

        self.assertEqual(len(layout.walls), 2)
        self.assertEqual(len(layout.rooms), 1)

        room = layout.rooms[0]
        self.assertEqual(room.id, "room_0")
        self.assertEqual(room.type, "LivingRoom")
        self.assertEqual(len(room.wall_ids), 2)
        self.assertIn("wall_0", room.wall_ids)
        self.assertIn("wall_1", room.wall_ids)

    def test_layout_with_special_tokens(self):
        # Format <int> means values are wrapped in < >
        layout_str = """
wall_0=(<0>, <0>, <0>, <10>, <0>, <0>, <3>, <0.2>)
"""
        layout = Layout(layout_str)

        self.assertEqual(len(layout.walls), 1)
        wall = layout.walls[0]
        self.assertAlmostEqual(float(wall.ax), 0)
        self.assertAlmostEqual(float(wall.bx), 10)
        self.assertAlmostEqual(float(wall.height), 3)

    def test_layout_with_room_quoted_walls(self):
        # Testing different quote styles in room definition
        layout_str = """
wall_0=(0, 0, 0, 10, 0, 0, 3, 0.2)
room_0=(['wall_0'], "Bedroom")
"""
        layout = Layout(layout_str)
        self.assertEqual(len(layout.rooms), 1)
        room = layout.rooms[0]
        self.assertEqual(room.type, "Bedroom")
        self.assertEqual(room.wall_ids[0], "wall_0")


if __name__ == "__main__":
    unittest.main()
