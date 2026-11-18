"""CSC376 project: a Panda builds a block bridge between two riverbanks.

Workflow
1. Enter the coordinates of the two banks' inner-edge midpoints (Panda base
   frame; z is the bank top height).
2. From the preset block size, compute automatically:
   - how many equally spaced piers are needed (spacing = one deck block length);
   - how many layers each pier has (so the pier top is as level with the banks
     as possible), with two parallel blocks per layer, each layer rotated 90°;
   - the placement pose and build order of every block.
3. For each block plan: a joint-space transfer at a safe height, Cartesian
   vertical descent/lift, and the gripper actions.
4. Print the layout, the checks and the total duration (the project requires
   finishing within 5 minutes).
5. With DRY_RUN=False: preview the whole build in Swift first, then type yes
   once and the real robot executes it.

Every parameter marked "MEASURE" must be measured at the station and updated.
"""

from dataclasses import dataclass
import math

import numpy as np
import roboticstoolbox as rtb
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from spatialmath import SE3

from csc376_franky.motion_generator import RuckigMotionGenerator


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Block size [length, width, height] in metres. MEASURE.
BLOCK_L, BLOCK_W, BLOCK_H = 0.15, 0.05, 0.03

# Inner-edge midpoints of the two banks, Panda base frame [x, y, z],
# z = bank top height. MEASURE.
# How: jog the fingertips to the bank edge and read robot.get_current_pose().t.
DEFAULT_BANK_A = np.array([0.45, -0.12, 0.09])
DEFAULT_BANK_B = np.array([0.45, 0.12, 0.09])

# Table height in the base frame. MEASURE.
TABLE_Z = 0.0

# Supply positions: blocks lie flat on the table, (x, y, yaw of the long side).
# TO DECIDE: must be reachable and out of the way of the bridge.
SUPPLY = [
    (0.33, 0.36, math.pi / 2),
    (0.42, 0.36, math.pi / 2),
    (0.51, 0.36, math.pi / 2),
    (0.60, 0.36, math.pi / 2),
    (0.33, -0.36, math.pi / 2),
    (0.42, -0.36, math.pi / 2),
    (0.51, -0.36, math.pi / 2),
    (0.60, -0.36, math.pi / 2),
]

MIN_BANK_BEARING = 0.02  # minimum length of deck resting on each bank
DECK_GAP = 0.004         # gap between adjacent deck ends, so placing the second one does not hit the first
FINGER_CLEARANCE = 0.02  # minimum gap between the two blocks of a layer (the fingers go in there)
HEIGHT_TOLERANCE = 0.005 # allowed height difference between pier top and bank top
TRANSFER_MARGIN = 0.05   # clearance of the carried block's bottom above the built structure
PLACE_DROP = 0.002       # how far above the target the TCP stops before releasing

# Gripper (the Franka Hand opens at most 0.08 m)
OPEN_WIDTH = min(BLOCK_W + 0.02, 0.08)
GRIPPER_SPEED = 0.1
GRASP_FORCE = 20.0
GRASP_TOLERANCE = 0.005

# Speeds: vertical segments use the course's Cartesian planner, transfers use
# joint-space Ruckig.
CART_VEL, CART_ACC, CART_JERK = 0.05, 0.05, 0.05
JOINT_VEL_FACTOR = 0.25  # the miniPC's limit is 0.3
JOINT_ACC_FACTOR = 0.3
TIME_LIMIT = 300.0

# True: only plan and print; no Swift, no real robot.
DRY_RUN = False
RUN_ON_REAL = False

# Panda joint limits (the datasheet values the miniPC checks against)
JOINT_VEL_LIM = np.array([2.175] * 4 + [2.61] * 3)
JOINT_ACC_LIM = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
JOINT_JERK_LIM = JOINT_ACC_LIM * 500.0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    name: str
    center: np.ndarray  # block centre [x, y, z]
    yaw: float          # direction of the block's long side


def wrap_yaw(yaw: float) -> float:
    """A block looks the same after a 180° turn about the vertical axis, so keep
    yaw in (-π/2, π/2] to minimise how far joint 7 has to turn."""

    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if math.isclose(yaw, -math.pi / 2) else yaw


def plan_layout(bank_a: np.ndarray, bank_b: np.ndarray):
    """Return (Placements in build order, report lines, warnings, structure top height)."""

    report, warnings = [], []
    axis = bank_b[:2] - bank_a[:2]
    span = float(np.linalg.norm(axis))
    u = np.append(axis / span, 0.0)          # along the bridge
    v = np.array([-u[1], u[0], 0.0])         # across the bridge
    axis_yaw = math.atan2(u[1], u[0])
    bank_z = (bank_a[2] + bank_b[2]) / 2

    if abs(bank_a[2] - bank_b[2]) > HEIGHT_TOLERANCE:
        warnings.append(f"The banks differ in height by {abs(bank_a[2] - bank_b[2]) * 100:.1f} cm; the deck will tilt")

    # Deck blocks are laid end to end:
    # (n_piers + 1)·L + n_piers·gap = span + 2·bank_bearing
    n_decks = math.ceil((span + 2 * MIN_BANK_BEARING + DECK_GAP) / (BLOCK_L + DECK_GAP))
    n_decks = max(n_decks, 1)
    n_piers = n_decks - 1
    bank_bearing = (n_decks * BLOCK_L + n_piers * DECK_GAP - span) / 2
    report.append(f"Span {span * 100:.1f} cm -> {n_decks} deck block(s), {n_piers} pier(s)")
    report.append(f"Deck rests {bank_bearing * 100:.1f} cm on each bank")
    if bank_bearing > BLOCK_L / 2:
        warnings.append("The deck rests more than half a block on the banks: the span is short; use fewer piers or move the banks closer")

    # Make the pier top as level with the banks as possible
    pier_base = TABLE_Z
    n_layers = max(1, round((bank_z - pier_base) / BLOCK_H))
    pier_top = pier_base + n_layers * BLOCK_H
    report.append(
        f"Bank top {bank_z * 100:.1f} cm, pier {n_layers} layer(s), pier top {pier_top * 100:.1f} cm"
    )
    if abs(pier_top - bank_z) > HEIGHT_TOLERANCE:
        warnings.append(
            f"Pier top differs from bank top by {(pier_top - bank_z) * 100:+.1f} cm: make the banks "
            f"{pier_top * 100:.1f} cm high, or add shims"
        )

    # Two parallel blocks per layer, outer faces flush with the pier's L×L square footprint
    offset = BLOCK_L / 2 - BLOCK_W / 2
    gap = 2 * offset - BLOCK_W
    report.append(f"Gap between the two blocks of a layer: {gap * 100:.1f} cm")
    if gap < FINGER_CLEARANCE:
        warnings.append(f"Layer gap {gap * 100:.1f} cm is less than the {FINGER_CLEARANCE * 100:.1f} cm the fingers need")

    placements: list[Placement] = []
    start = bank_a - u * bank_bearing  # where the first deck block starts (on bank A)
    start[2] = bank_z
    for k in range(n_piers):
        pier_center = start + u * ((k + 1) * BLOCK_L + (k + 0.5) * DECK_GAP)
        for layer in range(n_layers):
            # The top layer runs across the bridge axis, so each deck end rests on its own cross block
            across = (n_layers - 1 - layer) % 2 == 0
            dirn = v if across else u     # direction of the blocks' long side
            side = u if across else v     # direction the two blocks are offset in
            z = pier_base + (layer + 0.5) * BLOCK_H
            for s, tag in ((-1, "a"), (1, "b")):
                c = pier_center + side * (s * offset)
                c[2] = z
                yaw = wrap_yaw(math.atan2(dirn[1], dirn[0]))
                placements.append(Placement(f"pier{k + 1}-L{layer + 1}{tag}", c, yaw))

    deck_z = pier_top + BLOCK_H / 2
    for i in range(n_decks):
        c = start + u * (i * (BLOCK_L + DECK_GAP) + BLOCK_L / 2)
        c[2] = deck_z
        placements.append(Placement(f"deck{i + 1}", c, wrap_yaw(axis_yaw)))

    if len(placements) > len(SUPPLY):
        warnings.append(f"{len(placements)} blocks are needed but only {len(SUPPLY)} supply positions are defined")

    return placements, report, warnings, max(bank_z, pier_top)


# ---------------------------------------------------------------------------
# Motion planning
# ---------------------------------------------------------------------------

def tcp_pose(position: np.ndarray, yaw: float) -> SE3:
    """Gripper pointing straight down; the fingers close perpendicular to yaw,
    i.e. across the block's width."""

    return SE3.Trans(position) * SE3.Rz(yaw) * SE3.Rx(math.pi)


def solve_ik(panda: rtb.Robot, pose: SE3, seed: np.ndarray) -> np.ndarray:
    """Joint positions for a waypoint, kept close to a standard elbow-up configuration.

    The Panda has 7 joints (it is redundant). Solving IK only from the previous
    pose lets the configuration drift into odd ones (joints 1 and 3 cancelling
    each other); after that, even a purely vertical end-effector move turns the
    joints fast enough for the miniPC to refuse it. So seed with "qr turned
    towards the target" and keep the solution closest to that configuration.
    """

    nominal = np.array(panda.qr, dtype=float)
    nominal[0] = math.atan2(pose.t[1], pose.t[0])
    seeds = []
    for sign in (1, -1):
        s = nominal.copy()
        yaw = math.atan2(pose.R[1, 0], pose.R[0, 0])
        s[6] = np.clip(math.pi / 4 + sign * (nominal[0] - yaw), -2.8, 2.8)
        seeds.append(s)
    seeds.append(np.asarray(seed, dtype=float))

    best = None
    for s in seeds:
        sol = panda.ikine_LM(pose, q0=s, tol=1e-12, method="sugihara", k=1e-4, slimit=1)
        if not sol.success:
            continue
        cost = np.linalg.norm(sol.q[:6] - nominal[:6])
        if best is None or cost < best[0]:
            best = (cost, np.asarray(sol.q))
    if best is None:
        raise RuntimeError(f"No IK solution for {np.round(pose.t, 3)}: the target may be out of reach")
    return best[1]


def joint_trajectory(q0: np.ndarray, q1: np.ndarray, dt: float = 0.01):
    """Joint-space Ruckig trajectory, used for long transfers (avoids the
    problems of interpolating rotation vectors)."""

    otg = Ruckig(7, dt)
    inp, out = InputParameter(7), OutputParameter(7)
    inp.current_position = list(q0)
    inp.current_velocity = [0.0] * 7
    inp.current_acceleration = [0.0] * 7
    inp.target_position = list(q1)
    inp.target_velocity = [0.0] * 7
    inp.target_acceleration = [0.0] * 7
    inp.max_velocity = list(JOINT_VEL_LIM * JOINT_VEL_FACTOR)
    inp.max_acceleration = list(JOINT_ACC_LIM * JOINT_ACC_FACTOR)
    inp.max_jerk = list(JOINT_JERK_LIM * JOINT_ACC_FACTOR)
    traj = [np.array(q0, dtype=float)]
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        if res not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig joint planning failed: {res}")
        traj.append(np.array(out.new_position))
        out.pass_to_input(inp)
    return traj, dt


def vertical_trajectory(panda, gen: RuckigMotionGenerator, q0: np.ndarray, z: float):
    """Move straight up or down to height z, keeping the current orientation
    (a Cartesian straight line, so the block goes in and out vertically).

    The target orientation is taken from the start pose's fkine itself, so the
    two rotation vectors cannot differ in sign and flip the interpolation.
    """

    start = panda.fkine(q0)
    target = SE3.Rt(start.R, np.array([start.t[0], start.t[1], z]))
    cart, dt = gen.calculate_cartesian_pose_trajectory(
        start, target, CART_VEL, CART_ACC, CART_JERK
    )
    return gen.cartesian_pose_to_joint_trajectory(panda, q0, cart), dt


@dataclass
class Step:
    label: str
    kind: str                 # "move" | "open" | "grasp"
    q_traj: list | None = None
    dt: float = 0.0
    width: float = 0.0

    @property
    def duration(self) -> float:
        if self.kind == "move":
            return (len(self.q_traj) - 1) * self.dt
        return 1.0  # rough estimate for a gripper action; MEASURE


def plan_build(panda, placements, structure_top, q_start):
    gen = RuckigMotionGenerator()
    safe_z = structure_top + BLOCK_H + TRANSFER_MARGIN  # the TCP is at the block's mid-height
    steps: list[Step] = [Step("open gripper", "open", width=OPEN_WIDTH)]
    q = np.asarray(q_start, dtype=float)

    for placement, (sx, sy, syaw) in zip(placements, SUPPLY):
        pick_z = TABLE_Z + BLOCK_H / 2
        pick_yaw = wrap_yaw(syaw)
        name = placement.name

        q_above = solve_ik(panda, tcp_pose(np.array([sx, sy, safe_z]), pick_yaw), q)
        traj, dt = joint_trajectory(q, q_above)
        steps.append(Step(f"{name}: move above supply", "move", traj, dt))
        q = traj[-1]

        traj, dt = vertical_trajectory(panda, gen, q, pick_z)
        steps.append(Step(f"{name}: descend to pick", "move", traj, dt))
        q = traj[-1]
        steps.append(Step(f"{name}: grasp", "grasp", width=BLOCK_W))

        traj, dt = vertical_trajectory(panda, gen, q, safe_z)
        steps.append(Step(f"{name}: lift", "move", traj, dt))
        q = traj[-1]

        above = placement.center.copy()
        above[2] = safe_z
        q_above = solve_ik(panda, tcp_pose(above, placement.yaw), q)
        traj, dt = joint_trajectory(q, q_above)
        steps.append(Step(f"{name}: transfer above place pose", "move", traj, dt))
        q = traj[-1]

        traj, dt = vertical_trajectory(panda, gen, q, placement.center[2] + PLACE_DROP)
        steps.append(Step(f"{name}: descend to place", "move", traj, dt))
        q = traj[-1]
        steps.append(Step(f"{name}: release", "open", width=OPEN_WIDTH))

        traj, dt = vertical_trajectory(panda, gen, q, safe_z)
        steps.append(Step(f"{name}: retreat", "move", traj, dt))
        q = traj[-1]

    return steps


def check_steps(steps):
    """Pre-check against the miniPC's rules, to catch trajectories it would refuse."""

    problems = []
    for st in steps:
        if st.kind != "move":
            continue
        q = np.asarray(st.q_traj)
        step = np.abs(np.diff(q, axis=0))
        if step.max() > 0.025:
            problems.append(f"{st.label}: joint jump of {step.max():.3f} rad between points (IK flip?)")
        vel = step / st.dt
        if (vel > 0.3 * JOINT_VEL_LIM).any():
            problems.append(f"{st.label}: joint velocity above the miniPC limit")
    return problems


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_steps(steps, visualizer=None, real_robot=None):
    for st in steps:
        print(("[real] " if real_robot else "[preview] ") + st.label)
        if st.kind == "move":
            if real_robot:
                real_robot.run_joint_trajectory(st.q_traj, st.dt)
            else:
                visualizer.run_joint_trajectory(st.q_traj, st.dt)
        elif real_robot is None:
            visualizer.move_gripper(st.width, GRIPPER_SPEED)
        elif st.kind == "open":
            real_robot.gripper.move(st.width, GRIPPER_SPEED)
        else:
            ok = real_robot.gripper.grasp(
                width=BLOCK_W,
                speed=GRIPPER_SPEED,
                force=GRASP_FORCE,
                epsilon_inner=GRASP_TOLERANCE,
                epsilon_outer=GRASP_TOLERANCE,
            )
            if not ok:
                raise RuntimeError(f"{st.label}: the gripper did not grasp the block; stopping the build")


def read_point(prompt: str, default: np.ndarray) -> np.ndarray:
    text = input(f"{prompt} x y z (metres, Enter for default {default.tolist()}): ").strip()
    if not text:
        return default.copy()
    values = [float(t) for t in text.replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError("Expected 3 numbers: x y z")
    return np.array(values)


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    bank_a = read_point("Bank A inner-edge midpoint", DEFAULT_BANK_A)
    bank_b = read_point("Bank B inner-edge midpoint", DEFAULT_BANK_B)

    placements, report, warnings, top = plan_layout(bank_a, bank_b)
    print("\n== Layout ==")
    for line in report:
        print(" ", line)
    for p in placements:
        print(f"  {p.name:<10} centre {p.center}  yaw {math.degrees(p.yaw):6.1f}°")
    for w in warnings:
        print("  [warning]", w)

    panda = rtb.models.Panda()
    real_robot = None
    if RUN_ON_REAL and not DRY_RUN:
        from csc376_franky import FrankaJointTrajectoryController

        real_robot = FrankaJointTrajectoryController()
        q_start = real_robot.get_current_joint_positions()
    else:
        q_start = panda.qr

    print("\n== Motion plan ==")
    steps = plan_build(panda, placements, top, q_start)
    total = sum(st.duration for st in steps)
    n_blocks = min(len(placements), len(SUPPLY))
    print(f"  {n_blocks} blocks, {len(steps)} steps, about {total:.0f} s (limit {TIME_LIMIT:.0f} s)")
    for problem in check_steps(steps):
        print("  [problem]", problem)
    if total > TIME_LIMIT:
        print("  [problem] Too slow: raise JOINT_VEL_FACTOR / CART_VEL, or lower the safe height")

    if DRY_RUN:
        return

    from csc376_franky.visualizer import RtbVisualizer

    visualizer = RtbVisualizer(panda, q_start, real_robot=real_robot)
    try:
        input("Press Enter to preview the whole build in Swift.")
        run_steps(steps, visualizer=visualizer)
        if real_robot is None:
            return
        if input("Preview done. When it is safe, type yes to run it on the real robot: ").strip().lower() != "yes":
            print("You did not type yes; the real robot will not move.")
            return
        # The real robot must start from the pose the preview started from:
        # do not touch the robot during the preview.
        run_steps(steps, real_robot=real_robot)
        print("Build complete; the robot has withdrawn.")
    finally:
        visualizer.stop()
        if real_robot is not None:
            real_robot.close()


if __name__ == "__main__":
    main()
