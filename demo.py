"""One real grasp, with the same settings as proj.py.

The Panda picks up one block from a supply position and lifts it, using exactly
the parameters (block size, table height, supply positions, gripper and speed
settings) and planning functions of proj.py. Use it to test a single grasp
before running the whole bridge build.

Switches (same meaning as in proj.py)
- DRY_RUN = True: only plan and print.
- DRY_RUN = False, RUN_ON_REAL = False: preview in Swift, with the block drawn
  and carried by the gripper once it is grasped.
- DRY_RUN = False, RUN_ON_REAL = True: preview, type yes once, then the real
  robot runs the same steps.
"""

import math

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

import proj
from proj import (
    BLOCK_H,
    BLOCK_L,
    BLOCK_W,
    GRASP_DEPTH,
    GRIP_WIDTH,
    OPEN_WIDTH,
    Step,
    check_steps,
    grip_yaw,
    joint_trajectory,
    run_steps,
    solve_ik,
    supply_picks,
    tcp_pose,
    vertical_trajectory,
    wrap_yaw,
)
from csc376_franky.motion_generator import RuckigMotionGenerator


# ---------------------------------------------------------------------------
# Parameters (everything else comes from proj.py)
# ---------------------------------------------------------------------------

# Which block of proj.supply_picks() to pick: 0 is the top block of the first
# supply stack (the first one proj.py picks). The blocks below it in the same
# stack are drawn too.
PICK_INDEX = 0

# Approach from this far above the block's top, and lift this far after grasping.
APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.10

DRY_RUN = False
RUN_ON_REAL = True


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_grasp(panda, q_start):
    """Steps for one grasp: open, move above, descend, grasp, lift."""

    gen = RuckigMotionGenerator()
    sx, sy, syaw, top_z = supply_picks()[PICK_INDEX]
    yaw = grip_yaw(syaw)
    pick_z = top_z - GRASP_DEPTH  # same grasp height as proj.py
    above_z = top_z + APPROACH_HEIGHT

    steps = [Step("open gripper", "open", width=OPEN_WIDTH)]
    q = np.asarray(q_start, dtype=float)

    q_above = solve_ik(panda, tcp_pose(np.array([sx, sy, above_z]), yaw), q)
    traj, dt = joint_trajectory(q, q_above)
    steps.append(Step("move above the block", "move", traj, dt))
    q = traj[-1]

    traj, dt = vertical_trajectory(panda, gen, q, pick_z)
    steps.append(Step("descend to pick", "move", traj, dt))
    q = traj[-1]
    steps.append(Step("grasp", "grasp", width=GRIP_WIDTH))

    traj, dt = vertical_trajectory(panda, gen, q, pick_z + LIFT_HEIGHT)
    steps.append(Step("lift", "move", traj, dt))

    block_pose = SE3.Trans(sx, sy, top_z - BLOCK_H / 2) * SE3.Rz(wrap_yaw(syaw))
    return steps, block_pose


def stack_below(block_pose):
    """Poses of the blocks under the picked one in its stack (they stay put)."""

    sx, sy, syaw, top_z = supply_picks()[PICK_INDEX]
    poses = []
    for other in supply_picks()[PICK_INDEX + 1:]:
        if other[:3] != (sx, sy, syaw):
            break
        poses.append(SE3.Trans(0, 0, other[3] - top_z) * block_pose)
    return poses


# ---------------------------------------------------------------------------
# Swift preview with a block that follows the gripper
# ---------------------------------------------------------------------------

def add_block(visualizer, pose, color=(0.9, 0.6, 0.2)):
    """Draw a block in Swift. Returns None if this Swift cannot add shapes."""

    try:
        import spatialgeometry as sg

        block = sg.Cuboid(scale=[BLOCK_L, BLOCK_W, BLOCK_H], pose=pose, color=color)
        visualizer.env.add(block)
        return block
    except Exception as error:
        print(f"Could not draw the block in Swift ({error}); previewing without it.")
        return None


def preview(steps, visualizer, panda, block):
    """Run the steps in Swift. After the grasp the block moves with the gripper."""

    grip = {"offset": None}

    def follow(_index):
        if block is not None and grip["offset"] is not None:
            block.T = (panda.fkine(visualizer.q) * grip["offset"]).A

    visualizer.set_trajectory_callback(follow)
    for st in steps:
        print("[preview] " + st.label)
        if st.kind == "move":
            visualizer.run_joint_trajectory(st.q_traj, st.dt)
        else:
            visualizer.move_gripper(st.width, proj.GRIPPER_SPEED)
            if st.kind == "grasp" and block is not None:
                # Keep the block where it is relative to the TCP from now on.
                grip["offset"] = panda.fkine(visualizer.q).inv() * SE3(block.T)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    panda = rtb.models.Panda()

    real_robot = None
    if RUN_ON_REAL and not DRY_RUN:
        from csc376_franky import FrankaJointTrajectoryController

        real_robot = FrankaJointTrajectoryController()
        q_start = real_robot.get_current_joint_positions()
    else:
        q_start = panda.qr

    steps, block_pose = plan_grasp(panda, q_start)
    total = sum(st.duration for st in steps)
    sx, sy, syaw, top_z = supply_picks()[PICK_INDEX]
    print(f"Block {BLOCK_L * 100:.1f} x {BLOCK_W * 100:.1f} x {BLOCK_H * 100:.1f} cm, "
          f"gripped across {GRIP_WIDTH * 100:.1f} cm (opening {OPEN_WIDTH * 100:.1f} cm)")
    print(f"Pick block {PICK_INDEX}: ({sx:.3f}, {sy:.3f}), top at z={top_z:.3f}, yaw {math.degrees(syaw):.0f}°")
    print(f"{len(steps)} steps, about {total:.1f} s")
    problems = check_steps(steps)
    for problem in problems:
        print("  [problem]", problem)
    if DRY_RUN:
        return
    if problems:
        print("The miniPC would refuse this plan; fix the problems first.")
        return

    from csc376_franky.visualizer import RtbVisualizer

    visualizer = RtbVisualizer(panda, q_start, real_robot=real_robot)
    try:
        block = add_block(visualizer, block_pose)
        for pose in stack_below(block_pose):
            add_block(visualizer, pose, color=(0.6, 0.6, 0.6))
        input("Press Enter to preview the grasp in Swift.")
        preview(steps, visualizer, panda, block)
        if real_robot is None:
            input("Preview done. Press Enter to close Swift.")
            return
        if input("Preview done. When it is safe, type yes to run it on the real robot: ").strip().lower() != "yes":
            print("You did not type yes; the real robot will not move.")
            return
        # The real robot must start from the pose the preview started from:
        # do not touch the robot during the preview.
        run_steps(steps, real_robot=real_robot)
        print("Grasp complete; the block is held in the gripper.")
        input("Press Enter to close Swift (the gripper is not opened).")
    finally:
        visualizer.stop()
        if real_robot is not None:
            real_robot.close()


if __name__ == "__main__":
    main()
