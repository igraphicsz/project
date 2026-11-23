"""Read-only helper for measuring the supply stack on the real robot.

It never moves the robot or the gripper: it only reads the TCP pose.

Use
1. Activate FCI in Desk (end effector: Franka Hand). Put the robot in hand-guiding
   mode and move the gripper, pointing straight down, until the fingertips are
   centred over the block you want to measure (the fingers around its long faces,
   as proj.py grips it).
2. Press Enter here: it prints where the TCP is, how that compares with the
   centre of the top block in proj.SUPPLY_STACKS, and a SUPPLY_STACKS line to copy.
3. q + Enter to quit.

Run:  python debug.py
"""

import math

import numpy as np

import proj


def gripper_yaw_and_tilt(T):
    """Yaw of the gripper (proj.tcp_pose's convention) and its tilt from vertical [deg]."""

    R = T.R
    yaw = math.atan2(R[1, 0], R[0, 0])
    tilt = math.degrees(math.acos(np.clip(-R[2, 2], -1.0, 1.0)))  # 0 = pointing straight down
    return yaw, tilt


def block_yaw_from_gripper(yaw):
    """The block yaw (direction of its long side) this gripper yaw would grip."""

    if proj.GRIP_ACROSS == "length":
        yaw -= math.pi / 2
    # A block looks the same turned by 180°: report it in (-90°, 90°].
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return yaw if yaw > -math.pi / 2 + 1e-9 else yaw + math.pi


def report(T, gripper):
    p = T.t
    yaw, tilt = gripper_yaw_and_tilt(T)
    byaw = block_yaw_from_gripper(yaw)

    print("\n== TCP (fingertip centre), base frame ==")
    print(f"  x = {p[0]:.4f}  y = {p[1]:.4f}  z = {p[2]:.4f}  [m]")
    print(f"  gripper yaw {math.degrees(yaw):7.1f}°  -> block long side at {math.degrees(byaw):6.1f}°")
    print(f"  tilt from vertical {tilt:.1f}°" + ("   [warning] tilted: x, y are off, straighten it" if tilt > 3 else ""))
    if gripper is not None:
        try:
            st = gripper.state
            print(f"  gripper width {st.width * 100:.1f} cm, grasped: {st.is_grasped}")
        except Exception as error:
            print(f"  (could not read the gripper: {error})")

    print("\n== Compared with proj.py ==")
    print(f"  block {proj.BLOCK_L * 100:.1f} x {proj.BLOCK_W * 100:.1f} x {proj.BLOCK_H * 100:.1f} cm, "
          f"TABLE_Z = {proj.TABLE_Z:.4f}, GRASP_DEPTH = {proj.GRASP_DEPTH * 100:.2f} cm, grip across {proj.GRIP_ACROSS}")
    picks = proj.supply_picks()
    if not picks:
        print("  SUPPLY_STACKS is empty")
    else:
        for i, (sx, sy, syaw, count) in enumerate(proj.SUPPLY_STACKS):
            top_z = proj.TABLE_Z + count * proj.BLOCK_H
            pick_z = top_z - proj.GRASP_DEPTH
            d = p - np.array([sx, sy, pick_z])
            along = d[0] * math.cos(syaw) + d[1] * math.sin(syaw)    # along the block's long side
            across = -d[0] * math.sin(syaw) + d[1] * math.cos(syaw)  # along its width
            dyaw = (math.degrees(byaw - syaw) + 90) % 180 - 90  # a block is the same turned 180°
            print(f"  stack {i}: proj.py grasps its top block at ({sx:.4f}, {sy:.4f}, {pick_z:.4f}), yaw {math.degrees(syaw):.0f}°")
            print(f"    you are   dx = {d[0] * 1000:+6.1f}  dy = {d[1] * 1000:+6.1f}  dz = {d[2] * 1000:+6.1f} mm from it")
            print(f"    i.e.      {along * 1000:+6.1f} mm along the long side, {across * 1000:+6.1f} mm across it, "
                  f"yaw {dyaw:+.1f}°")

    print("\n== If the fingertips are now centred on a block ==")
    print(f"  SUPPLY_STACKS line:  ({p[0]:.4f}, {p[1]:.4f}, math.radians({math.degrees(byaw):.0f}), <number of blocks>),")
    print("  If the fingertips are at the top block's mid-height and the stack has n blocks:")
    for n in range(1, 6):
        table = p[2] + proj.GRASP_DEPTH - n * proj.BLOCK_H
        print(f"    n = {n}: TABLE_Z = {table:+.4f}")


def main():
    from csc376_franky import FrankaJointTrajectoryController

    np.set_printoptions(precision=4, suppress=True)
    robot = FrankaJointTrajectoryController()
    try:
        print("Connected. This script only reads; it does not move the robot.")
        while True:
            cmd = input("\nEnter = read the pose, q = quit: ").strip().lower()
            if cmd == "q":
                break
            report(robot.get_current_pose(), robot.gripper)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
