
import unittest
import sys
import os
import numpy as np

# Add repo root to path to import spatiallm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatiallm.layout.entity import Wall, Door, Bbox, Window
from spatiallm.layout.layout import Layout

class TestTokenString(unittest.TestCase):
    def setUp(self):
        self.num_bins = 1600
        
    def test_wall_token_string(self):
        # Create a wall with specific coordinates
        # We use values that are easy to reason about after normalization if possible
        # But here we just check if the method produces the expected format
        wall = Wall(id=1, ax=0, ay=0, az=0, bx=10, by=10, bz=0, height=3, thickness=0.2)
        
        # Mock the discretization by setting values directly to integers 
        # to avoid dependency on normalization constants for this specific test
        wall.ax = 0
        wall.ay = 0
        wall.az = 0
        wall.bx = 500
        wall.by = 500
        wall.bz = 0
        wall.height = 150
        wall.thickness = 10
        
        expected = "wall_1=Wall(<0>,<0>,<0>,<500>,<500>,<0>,<150>,<10>)"
        self.assertEqual(wall.to_token_string(), expected)

    def test_door_token_string(self):
        door = Door(id=1001, wall_id=1, position_x=5, position_y=5, position_z=1, width=1, height=2)
        
        # Set discretized values manually
        door.position_x = 250
        door.position_y = 250
        door.position_z = 50
        door.width = 60
        door.height = 120
        
        # Note: id % 1000 logic is inside to_token_string
        # 1001 % 1000 = 1
        expected = "door_1=Door(wall_1,<250>,<250>,<50>,<60>,<120>)"
        self.assertEqual(door.to_token_string(), expected)

    def test_window_token_string(self):
        window = Window(id=2002, wall_id=1, position_x=5, position_y=5, position_z=1, width=1, height=1)
        
        window.position_x = 250
        window.position_y = 250
        window.position_z = 50
        window.width = 60
        window.height = 60
        
        # 2002 % 1000 = 2
        expected = "window_2=Window(wall_1,<250>,<250>,<50>,<60>,<60>)"
        self.assertEqual(window.to_token_string(), expected)

    def test_bbox_token_string(self):
        bbox = Bbox(id=3005, class_name="chair", position_x=2, position_y=2, position_z=0, 
                   angle_z=0, scale_x=1, scale_y=1, scale_z=1)
        
        bbox.position_x = 100
        bbox.position_y = 100
        bbox.position_z = 0
        bbox.angle_z = 10
        bbox.scale_x = 50
        bbox.scale_y = 50
        bbox.scale_z = 50
        
        # 3005 % 1000 = 5
        expected = "bbox_5=Bbox(chair,<100>,<100>,<0>,<10>,<50>,<50>,<50>)"
        self.assertEqual(bbox.to_token_string(), expected)

    def test_layout_token_string(self):
        # Create a layout with one of each entity
        layout = Layout()
        
        wall = Wall(id=0, ax=0, ay=0, az=0, bx=10, by=0, bz=0, height=3, thickness=0.2)
        wall.ax, wall.ay, wall.az = 0, 0, 0
        wall.bx, wall.by, wall.bz = 100, 0, 0
        wall.height, wall.thickness = 150, 10
        layout.walls.append(wall)
        
        door = Door(id=1, wall_id=0, position_x=5, position_y=0, position_z=1, width=1, height=2)
        door.position_x, door.position_y, door.position_z = 50, 0, 50
        door.width, door.height = 20, 100
        layout.doors.append(door)
        
        # Verify the combined string
        token_str = layout.to_token_string()
        lines = token_str.split('\n')
        
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "wall_0=Wall(<0>,<0>,<0>,<100>,<0>,<0>,<150>,<10>)")
        self.assertEqual(lines[1], "door_1=Door(wall_0,<50>,<0>,<50>,<20>,<100>)")

    def test_integration_with_discretization(self):
        # Test the full pipeline: create -> discretize -> to_token_string
        wall = Wall(id=10, ax=0.0, ay=0.0, az=0.0, bx=10.0, by=10.0, bz=0.0, height=2.56, thickness=0.2)
        
        # Using the normalization preset from entity.py:
        # world: (0.0, 32.0) -> 1600 bins. 0->0, 32->1599. 
        # 10.0 -> (10/32)*1600 = 500
        # height: (0.0, 25.6) -> 1600 bins. 2.56 -> (2.56/25.6)*1600 = 160
        
        wall.normalize_and_discretize(self.num_bins)
        
        token_str = wall.to_token_string()
        
        # Check if the output contains the expected token format
        self.assertIn("wall_10=Wall(", token_str)
        self.assertIn("<", token_str)
        self.assertIn(">", token_str)
        
        # Roughly check values (exact integer matching might be fragile due to float precision, 
        # but the logic above suggests 500 and 160)
        # Note: 0 maps to 0
        self.assertIn("<0>", token_str) # ax, ay, az, bz
        # 10.0 maps to 500
        self.assertIn("<500>", token_str) # bx, by
        # 2.56 height maps to 160
        self.assertIn("<160>", token_str) # height

if __name__ == '__main__':
    unittest.main()
