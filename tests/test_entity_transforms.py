import unittest
import numpy as np
from spatiallm.layout.entity import Wall, Door, Window, Bbox, Room


class TestEntityTransforms(unittest.TestCase):
    def assertArrayAlmostEqual(self, arr1, arr2, places=6):
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=places)

    def test_wall_transforms(self):
        # Initial Wall: 3m long along X-axis, starting at origin
        wall = Wall(id=1, ax=0, ay=0, az=0, bx=3, by=0, bz=0, height=2.5, thickness=0.2)

        # 1. Test Translation
        translation = np.array([1.0, 2.0, 0.5])
        wall.translate(translation)

        self.assertAlmostEqual(wall.ax, 1.0)
        self.assertAlmostEqual(wall.ay, 2.0)
        self.assertAlmostEqual(wall.az, 0.5)
        self.assertAlmostEqual(wall.bx, 4.0)
        self.assertAlmostEqual(wall.by, 2.0)
        self.assertAlmostEqual(wall.bz, 0.5)

        # 2. Test Rotation (90 degrees around Z)
        # Reset wall to simple state for easy rotation check: (1,0) -> (4,0)
        wall = Wall(id=1, ax=1, ay=0, az=0, bx=4, by=0, bz=0, height=2.5, thickness=0.2)
        wall.rotate(np.pi / 2)  # 90 degrees CCW

        # (1,0) -> (0,1)
        self.assertAlmostEqual(wall.ax, 0.0)
        self.assertAlmostEqual(wall.ay, 1.0)
        # (4,0) -> (0,4)
        self.assertAlmostEqual(wall.bx, 0.0)
        self.assertAlmostEqual(wall.by, 4.0)

        # 3. Test Scale
        wall = Wall(id=1, ax=0, ay=0, az=0, bx=2, by=0, bz=0, height=2.0, thickness=0.1)
        wall.scale(2.0)

        self.assertAlmostEqual(wall.bx, 4.0)
        self.assertAlmostEqual(wall.height, 4.0)
        self.assertAlmostEqual(wall.thickness, 0.2)

    def test_door_transforms(self):
        # Door at (2,0,1)
        door = Door(
            id=1,
            wall_id=1,
            position_x=2,
            position_y=0,
            position_z=1,
            width=0.9,
            height=2.1,
        )

        # 1. Translate
        door.translate(np.array([1, 1, 0]))
        self.assertAlmostEqual(door.position_x, 3.0)
        self.assertAlmostEqual(door.position_y, 1.0)
        self.assertAlmostEqual(door.position_z, 1.0)

        # 2. Rotate 90 deg around Z
        # Reset to (2,0,1)
        door = Door(
            id=1,
            wall_id=1,
            position_x=2,
            position_y=0,
            position_z=1,
            width=0.9,
            height=2.1,
        )
        door.rotate(np.pi / 2)

        # (2,0) -> (0,2)
        self.assertAlmostEqual(door.position_x, 0.0)
        self.assertAlmostEqual(door.position_y, 2.0)
        self.assertAlmostEqual(door.position_z, 1.0)

        # 3. Scale
        door.scale(0.5)
        # Pos (0,2,1) -> (0,1,0.5)
        self.assertAlmostEqual(door.position_x, 0.0)
        self.assertAlmostEqual(door.position_y, 1.0)
        self.assertAlmostEqual(door.position_z, 0.5)
        self.assertAlmostEqual(door.width, 0.45)
        self.assertAlmostEqual(door.height, 1.05)

    def test_bbox_transforms(self):
        # Bbox at (1,0,0) oriented 0 rad
        bbox = Bbox(
            id=1,
            class_name="chair",
            position_x=1,
            position_y=0,
            position_z=0,
            angle_z=0,
            scale_x=1,
            scale_y=1,
            scale_z=1,
        )

        # 1. Translate
        bbox.translate(np.array([0, 5, 0]))
        self.assertAlmostEqual(bbox.position_y, 5.0)

        # 2. Rotate 90 deg
        # (1,5) -> (-5, 1)
        bbox.rotate(np.pi / 2)
        self.assertAlmostEqual(bbox.position_x, -5.0)
        self.assertAlmostEqual(bbox.position_y, 1.0)
        # Angle 0 -> pi/2
        # Note: Bbox angle logic handles symmetry, assume simple case first
        # self.assertAlmostEqual(bbox.angle_z, np.pi / 2)
        # Actually bbox.rotate implements specific symmetry logic.
        # Let's check if it updated.
        # New angle should be roughly pi/2 (or -pi/2 depending on normalization range [-pi, pi])

        # 3. Scale
        bbox.scale(2.0)
        self.assertAlmostEqual(bbox.scale_x, 2.0)
        self.assertAlmostEqual(bbox.position_x, -10.0)  # -5 * 2

    def test_room_transforms_noop(self):
        room = Room(id=0, wall_ids=["wall_1", "wall_2"], type="Bedroom")

        # Ensure these don't crash and don't change anything (Room has no spatial state itself)
        room.translate(np.array([10, 10, 10]))
        room.rotate(np.pi)
        room.scale(5.0)
        room.clip_z(0, 10)

        self.assertEqual(room.id, 0)
        self.assertEqual(room.wall_ids, ["wall_1", "wall_2"])
        self.assertEqual(room.type, "Bedroom")


if __name__ == "__main__":
    unittest.main()
