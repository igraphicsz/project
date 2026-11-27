"""CSC376 project: a Panda builds a block bridge between two points.

Workflow
1. Enter the coordinates of the bridge's two end points (Panda base frame;
   z is the height the pier tops should reach). There are no riverbanks: a pier
   stands on each end point, plus equally spaced piers in between.
2. From the preset block size, compute automatically:
   - how many piers are needed and their spacing (wide enough for a finger to
     go between neighbouring piers, narrow enough for a deck block to rest on both);
   - how many layers each pier has, with two parallel blocks per layer, each
     layer rotated 90°;
   - the placement pose and build order of every block: all piers rise together,
     one layer at a time (layer 1 of every pier, then layer 2, ...), and the deck
     goes on last. A neighbouring pier is then never taller than the layer being
     placed, so the fingers never have to reach down beside a finished pier.
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
BLOCK_L, BLOCK_W, BLOCK_H = 0.075, 0.025, 0.015

# Which pair of faces the fingers close on:
# "width":  the two long faces, holding the block across BLOCK_W. A finger then
#           goes into the gap between the two blocks of a layer, so that gap
#           must be at least FINGER_CLEARANCE (needs BLOCK_L >= 2·BLOCK_W + 2 cm).
# "length": the two short end faces, holding the block across BLOCK_L. The two
#           blocks of a layer can then sit almost touching (LAYER_GAP apart).
#           Needs BLOCK_L + 2 cm <= 8 cm, what the Franka Hand opens to.
# With 7.5 cm blocks only "width" fits in the hand (7.5 + 2 > 8). The layer gap is
# then L − 2W = 2.5 cm, so both layers cover the same 7.5 cm square.
GRIP_ACROSS = "width"

# The bridge's two end points, Panda base frame [x, y, z]. A pier is built
# centred on each, and z is the height its top should reach (the table counts as
# z = TABLE_Z). Each pier gets round((z - TABLE_Z) / BLOCK_H) layers of 2 blocks:
# z = 0.03 with 1.5 cm blocks gives 2 layers, 4 blocks per pier.
DEFAULT_END_A = np.array([0.4681, -0.1399, 0.03])
DEFAULT_END_B = np.array([0.4656, -0.5945, 0.03])

# False: build only the piers, no deck blocks on top.
BUILD_DECK = False

# Table height in the base frame. MEASURE.
TABLE_Z = 0.0

# Supply stacks: blocks lie flat, stacked straight up on the table, all with the
# same yaw. (x, y, yaw of the long side, number of blocks in the stack).
# Blocks are taken from the top of the first stack until it is empty, then from
# the next stack. TO DECIDE: must be reachable and out of the way of the bridge.
SUPPLY_STACKS = [
    (-0.1427, 0.5232, math.pi / 2, 6),
    (-0.0208, 0.5232, math.pi / 2, 6),
    (0.1104, 0.5230, math.pi / 2, 6),
]

# How far below a block's top face the TCP (fingertip centre) goes to grasp it.
# Half a block grips it at mid-height. The finger pads reach below the TCP, so
# with a block underneath, keep this small enough that the pads stay above it.
# MEASURE with the real fingers: with thin blocks the pads may reach the block below.
GRASP_DEPTH = BLOCK_H / 2

MIN_DECK_BEARING = 0.02  # minimum length of a deck block resting on each pier it spans
FINGER_CLEARANCE = 0.02  # minimum gap wherever a finger has to go between two things (also between piers)
LAYER_GAP = 0.004        # minimum gap between the two blocks of a layer when no finger goes in it
HEIGHT_TOLERANCE = 0.005 # allowed difference between the pier top and the requested z
TRANSFER_MARGIN = 0.05   # clearance of the carried block's bottom above the built structure
PLACE_DROP = 0.002       # how far above the target the TCP stops before releasing

# Gripper (the Franka Hand opens at most 0.08 m)
GRIP_WIDTH = BLOCK_L if GRIP_ACROSS == "length" else BLOCK_W  # the size between the fingers
MAX_OPEN_WIDTH = 0.08
OPEN_WIDTH = min(GRIP_WIDTH + 0.02, MAX_OPEN_WIDTH)
# Opening when releasing a placed block. Kept small so the fingers do not hit the
# neighbouring blocks: each finger moves out (RELEASE_WIDTH - GRIP_WIDTH) / 2.
# The gripper reopens to OPEN_WIDTH only after retreating to the safe height.
# MEASURE: it must still let go of the block reliably.
RELEASE_WIDTH = min(GRIP_WIDTH + 0.006, OPEN_WIDTH)
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
RUN_ON_REAL = True

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


def pier_spacing(span: float, pier_half: float):
    """Number of intervals between the two end piers, and the warnings about it.

    Neighbouring piers must leave FINGER_CLEARANCE between them (a finger goes
    there when a cross layer is placed). With a deck, each deck block is centred
    over one gap and must rest MIN_DECK_BEARING on both piers, which caps the
    spacing; without a deck, use as few piers as the clearance allows.
    """

    min_d = 2 * pier_half + FINGER_CLEARANCE
    # A deck block centred between two piers rests (L - gap) / 2 on each.
    max_d = 2 * pier_half + BLOCK_L - 2 * MIN_DECK_BEARING
    warnings = []
    if span < min_d - 1e-9:
        warnings.append(
            f"The end points are only {span * 100:.1f} cm apart: two piers need at least "
            f"{min_d * 100:.1f} cm between centres; move the points apart"
        )
        return 1, warnings
    fewest = max(1, math.floor(span / min_d + 1e-9))
    if not BUILD_DECK:
        return fewest, warnings
    if max_d < min_d:
        warnings.append("A deck block is too short to span a gap the fingers fit into")
        return fewest, warnings
    n = math.ceil(span / max_d - 1e-9)
    if span / n < min_d - 1e-9:
        # No whole number of intervals fits this span: say which spans do, and
        # keep the finger clearance (the deck then rests a little less on each pier).
        warnings.append(
            f"No equal pier spacing between {min_d * 100:.1f} and {max_d * 100:.1f} cm fits a "
            f"{span * 100:.1f} cm span: make it at most {(n - 1) * max_d * 100:.1f} cm "
            f"or at least {n * min_d * 100:.1f} cm"
        )
        return fewest, warnings
    return n, warnings


def plan_layout(end_a: np.ndarray, end_b: np.ndarray):
    """Return (Placements in build order, report lines, warnings, structure top height)."""

    report, warnings = [], []
    axis = end_b[:2] - end_a[:2]
    span = float(np.linalg.norm(axis))
    if span < 1e-6:
        raise ValueError("The two end points are the same point")
    u = np.append(axis / span, 0.0)          # along the bridge
    v = np.array([-u[1], u[0], 0.0])         # across the bridge
    axis_yaw = math.atan2(u[1], u[0])
    target_z = (end_a[2] + end_b[2]) / 2

    if abs(end_a[2] - end_b[2]) > HEIGHT_TOLERANCE:
        warnings.append(
            f"The end points differ in height by {abs(end_a[2] - end_b[2]) * 100:.1f} cm; "
            f"all piers are built to the average, {target_z * 100:.1f} cm"
        )

    # Two parallel blocks per layer, offset ±offset from the pier centre. Gripped
    # across their width, a finger goes between them, so they sit flush with the
    # pier's L×L square and the gap is L − 2W. Gripped across their length, they
    # can sit LAYER_GAP apart.
    min_gap = FINGER_CLEARANCE if GRIP_ACROSS == "width" else LAYER_GAP
    offset = max(BLOCK_L / 2 - BLOCK_W / 2, BLOCK_W / 2 + min_gap / 2)
    gap = 2 * offset - BLOCK_W
    pier_half = max(BLOCK_L / 2, offset + BLOCK_W / 2)  # half the pier's footprint
    report.append(
        f"Gripped across the {GRIP_ACROSS} ({GRIP_WIDTH * 100:.1f} cm, opening "
        f"{OPEN_WIDTH * 100:.1f} cm, release {RELEASE_WIDTH * 100:.1f} cm); layer gap {gap * 100:.1f} cm; "
        f"pier footprint {pier_half * 200:.1f} cm square"
    )
    if GRIP_WIDTH + 0.02 > MAX_OPEN_WIDTH:
        warnings.append(
            f"The gripper must open to {(GRIP_WIDTH + 0.02) * 100:.1f} cm but opens to at most "
            f"{MAX_OPEN_WIDTH * 100:.0f} cm: grip across the other side"
        )
    if gap < min_gap - 1e-9:
        warnings.append(f"Layer gap {gap * 100:.1f} cm is less than the {min_gap * 100:.1f} cm needed")
    if offset > BLOCK_L / 2:
        warnings.append(
            "The blocks of a layer are farther apart than a block is long: "
            "the layer above does not reach their centres"
        )

    # A pier on each end point, the rest equally spaced in between.
    n_intervals, spacing_warnings = pier_spacing(span, pier_half)
    warnings += spacing_warnings
    spacing = span / n_intervals
    pier_centers = []
    for k in range(n_intervals + 1):
        c = end_a + u * (k * spacing)
        c[2] = target_z
        pier_centers.append(c)
    pier_gap = spacing - 2 * pier_half
    report.append(
        f"Span {span * 100:.1f} cm -> {len(pier_centers)} piers, {spacing * 100:.1f} cm apart "
        f"({pier_gap * 100:.1f} cm between neighbouring piers)"
    )

    pier_base = TABLE_Z
    n_layers = max(1, round((target_z - pier_base) / BLOCK_H))
    pier_top = pier_base + n_layers * BLOCK_H
    report.append(
        f"Requested height {target_z * 100:.1f} cm, pier {n_layers} layer(s), pier top {pier_top * 100:.1f} cm"
    )
    if abs(pier_top - target_z) > HEIGHT_TOLERANCE:
        warnings.append(
            f"Pier top differs from the requested height by {(pier_top - target_z) * 100:+.1f} cm "
            f"(z is rounded to whole layers of {BLOCK_H * 100:.1f} cm)"
        )

    # Layer by layer across all piers: layer 1 of every pier, then layer 2, ...
    # The pier order is reversed on every other layer so the arm does not have
    # to travel back to the first pier at the start of each layer.
    placements: list[Placement] = []
    for layer in range(n_layers):
        # The top layer runs across the bridge axis, so each deck end rests on its own cross block
        across = (n_layers - 1 - layer) % 2 == 0
        dirn = v if across else u     # direction of the blocks' long side
        side = u if across else v     # direction the two blocks are offset in
        z = pier_base + (layer + 0.5) * BLOCK_H
        yaw = wrap_yaw(math.atan2(dirn[1], dirn[0]))
        order = list(range(len(pier_centers)))
        if layer % 2 == 1:
            order.reverse()
        for k in order:
            for s, tag in ((-1, "a"), (1, "b")):
                c = pier_centers[k] + side * (s * offset)
                c[2] = z
                placements.append(Placement(f"pier{k + 1}-L{layer + 1}{tag}", c, yaw))

    # The deck goes on last: one block centred over each gap, along the bridge.
    top = pier_top
    if BUILD_DECK:
        bearing = (BLOCK_L - pier_gap) / 2
        report.append(f"{n_intervals} deck block(s), each resting {bearing * 100:.1f} cm on both piers")
        if bearing < 0.005:
            warnings.append(f"Deck blocks rest only {bearing * 100:.1f} cm on each pier: they will fall")
        elif bearing < MIN_DECK_BEARING - 1e-9:
            warnings.append(f"Deck blocks rest only {bearing * 100:.1f} cm on each pier (less than {MIN_DECK_BEARING * 100:.0f} cm)")
        deck_z = pier_top + BLOCK_H / 2
        for i in range(n_intervals):
            c = (pier_centers[i] + pier_centers[i + 1]) / 2
            c[2] = deck_z
            placements.append(Placement(f"deck{i + 1}", c, wrap_yaw(axis_yaw)))
        top = pier_top + BLOCK_H
    else:
        report.append("BUILD_DECK = False: piers only, no deck blocks")

    n_supply = sum(stack[3] for stack in SUPPLY_STACKS)
    report.append(f"{len(placements)} blocks needed, {n_supply} in the supply stacks")
    if len(placements) > n_supply:
        warnings.append(
            f"{len(placements)} blocks are needed but the supply stacks hold only {n_supply}: "
            f"add {len(placements) - n_supply} more to SUPPLY_STACKS (and to the table)"
        )
    if GRASP_DEPTH > BLOCK_H:
        warnings.append("GRASP_DEPTH is more than a block high: the fingers would reach the block below")

    return placements, report, warnings, top


# ---------------------------------------------------------------------------
# Motion planning
# ---------------------------------------------------------------------------

def grip_yaw(block_yaw: float) -> float:
    """Gripper yaw for a block with this yaw. tcp_pose closes the fingers
    perpendicular to its yaw, i.e. across BLOCK_W when it equals the block's
    yaw; to close them along the block's length the gripper turns 90°."""

    if GRIP_ACROSS == "length":
        return wrap_yaw(block_yaw + math.pi / 2)
    return wrap_yaw(block_yaw)


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


def supply_picks():
    """Where each supply block is, in the order they are picked.

    Returns a list of (x, y, yaw, top_z): top_z is the height of that block's
    top face. Each stack is emptied from the top down before the next one, so
    every pick is a little lower than the one before it on the same stack.
    """

    picks = []
    for sx, sy, syaw, count in SUPPLY_STACKS:
        for remaining in range(count, 0, -1):
            picks.append((sx, sy, syaw, TABLE_Z + remaining * BLOCK_H))
    return picks


def supply_top():
    """Height of the tallest supply stack, before any block is taken."""

    return TABLE_Z + max((stack[3] for stack in SUPPLY_STACKS), default=0) * BLOCK_H


def safe_height(structure_top):
    """TCP height for transfers: the carried block's bottom clears both the
    bridge and the supply stacks by TRANSFER_MARGIN."""

    tallest = max(structure_top, supply_top())
    # The TCP sits GRASP_DEPTH below the carried block's top, so the block's
    # bottom is BLOCK_H - GRASP_DEPTH below the TCP.
    return tallest + (BLOCK_H - GRASP_DEPTH) + TRANSFER_MARGIN


def plan_build(panda, placements, structure_top, q_start):
    gen = RuckigMotionGenerator()
    safe_z = safe_height(structure_top)
    steps: list[Step] = [Step("open gripper", "open", width=OPEN_WIDTH)]
    q = np.asarray(q_start, dtype=float)

    jobs = list(zip(placements, supply_picks()))
    for k, (placement, (sx, sy, syaw, top_z)) in enumerate(jobs):
        pick_z = top_z - GRASP_DEPTH
        pick_yaw = grip_yaw(syaw)
        name = placement.name

        q_above = solve_ik(panda, tcp_pose(np.array([sx, sy, safe_z]), pick_yaw), q)
        traj, dt = joint_trajectory(q, q_above)
        steps.append(Step(f"{name}: move above supply", "move", traj, dt))
        q = traj[-1]

        traj, dt = vertical_trajectory(panda, gen, q, pick_z)
        steps.append(Step(f"{name}: descend to pick", "move", traj, dt))
        q = traj[-1]
        steps.append(Step(f"{name}: grasp", "grasp", width=GRIP_WIDTH))

        traj, dt = vertical_trajectory(panda, gen, q, safe_z)
        steps.append(Step(f"{name}: lift", "move", traj, dt))
        q = traj[-1]

        above = placement.center.copy()
        above[2] = safe_z
        q_above = solve_ik(panda, tcp_pose(above, grip_yaw(placement.yaw)), q)
        traj, dt = joint_trajectory(q, q_above)
        steps.append(Step(f"{name}: transfer above place pose", "move", traj, dt))
        q = traj[-1]

        # The TCP holds the block GRASP_DEPTH below its top face, so for the
        # block's centre to land on placement.center the TCP stops this high.
        place_z = placement.center[2] + BLOCK_H / 2 - GRASP_DEPTH
        traj, dt = vertical_trajectory(panda, gen, q, place_z + PLACE_DROP)
        steps.append(Step(f"{name}: descend to place", "move", traj, dt))
        q = traj[-1]
        # Open only a little, so the fingers do not hit the neighbouring blocks.
        steps.append(Step(f"{name}: release", "open", width=RELEASE_WIDTH))

        traj, dt = vertical_trajectory(panda, gen, q, safe_z)
        steps.append(Step(f"{name}: retreat", "move", traj, dt))
        q = traj[-1]

        if RELEASE_WIDTH < OPEN_WIDTH and k < len(jobs) - 1:
            # Clear of everything now: open fully for the next pick.
            steps.append(Step(f"{name}: open for the next block", "open", width=OPEN_WIDTH))

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

class BuildScene:
    """Draws the end-point markers, the supply stacks and the blocks in Swift, and moves the
    blocks with the gripper during the preview. Preview only: none of this is
    sent to the real robot."""

    MARKER_COLOR = (0.9, 0.2, 0.2)
    SUPPLY_COLOR = (0.6, 0.6, 0.6)
    CARRIED_COLOR = (0.9, 0.6, 0.2)
    PLACED_COLOR = (0.2, 0.6, 0.9)
    MARKER_SIZE = 0.01  # side of the small cube drawn above each end point

    def __init__(self, panda, visualizer, end_a, end_b, placements):
        import spatialgeometry as sg

        self._sg = sg
        self.panda = panda
        self.visualizer = visualizer
        self.placements = placements
        self.blocks = []     # one per supply pick, in pick order
        self.n_grasped = 0
        self.carried = None  # (block, offset from the TCP) while held
        try:
            self._add_markers(end_a, end_b)
            for sx, sy, syaw, top_z in supply_picks():
                pose = SE3.Trans(sx, sy, top_z - BLOCK_H / 2) * SE3.Rz(wrap_yaw(syaw))
                self.blocks.append(self._box([BLOCK_L, BLOCK_W, BLOCK_H], pose, self.SUPPLY_COLOR))
            visualizer.set_trajectory_callback(self._follow)
            self.ok = True
        except Exception as error:
            print(f"Could not draw the blocks in Swift ({error}); previewing the robot only.")
            self.ok = False

    def _box(self, scale, pose, color):
        box = self._sg.Cuboid(scale=scale, pose=pose, color=color)
        self.visualizer.env.add(box)
        return box

    def _add_markers(self, end_a, end_b):
        """A small cube floating above each end point's requested height."""

        for point in (end_a, end_b):
            pose = SE3.Trans(point[0], point[1], point[2] + BLOCK_H + self.MARKER_SIZE)
            self._box([self.MARKER_SIZE] * 3, pose, self.MARKER_COLOR)

    def _tcp(self):
        return self.panda.fkine(self.visualizer.q)

    def _follow(self, _index):
        if self.carried is not None:
            block, offset = self.carried
            block.T = (self._tcp() * offset).A

    def after_step(self, st):
        """Update the blocks after a gripper step of the preview."""

        if not self.ok:
            return
        if st.kind == "grasp" and self.n_grasped < len(self.blocks):
            block = self.blocks[self.n_grasped]
            self.n_grasped += 1
            block.color = self.CARRIED_COLOR
            self.carried = (block, self._tcp().inv() * SE3(block.T))
        elif st.kind == "open" and self.carried is not None:
            # Released PLACE_DROP above the target: let it drop straight down.
            block, _ = self.carried
            block.T = (SE3.Trans(0, 0, -PLACE_DROP) * SE3(block.T)).A
            block.color = self.PLACED_COLOR
            self.carried = None


def run_steps(steps, visualizer=None, real_robot=None, scene=None):
    for st in steps:
        print(("[real] " if real_robot else "[preview] ") + st.label)
        if st.kind == "move":
            if real_robot:
                real_robot.run_joint_trajectory(st.q_traj, st.dt)
            else:
                visualizer.run_joint_trajectory(st.q_traj, st.dt)
        elif real_robot is None:
            visualizer.move_gripper(st.width, GRIPPER_SPEED)
            if scene is not None:
                scene.after_step(st)
        elif st.kind == "open":
            real_robot.gripper.move(st.width, GRIPPER_SPEED)
        else:
            ok = real_robot.gripper.grasp(
                width=GRIP_WIDTH,
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
    end_a = read_point("Bridge end point A", DEFAULT_END_A)
    end_b = read_point("Bridge end point B", DEFAULT_END_B)

    placements, report, warnings, top = plan_layout(end_a, end_b)
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
    n_blocks = min(len(placements), len(supply_picks()))
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
        scene = BuildScene(panda, visualizer, end_a, end_b, placements)
        input("Press Enter to preview the whole build in Swift.")
        run_steps(steps, visualizer=visualizer, scene=scene)
        if real_robot is None:
            input("Preview done. Press Enter to close Swift.")
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
