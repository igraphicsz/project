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

# Inner-edge midpoints of the two banks, Panda base frame [x, y, z],
# z = bank top height. MEASURE.
# How: jog the fingertips to the bank edge and read robot.get_current_pose().t.
# The bridge runs along y. The pier gets round(bank height / BLOCK_H) layers of
# 2 blocks: z = 0.03 with 1.5 cm blocks gives 2 layers, 4 blocks.
# Without a deck the single pier sits midway between the banks, so the span must
# leave FINGER_CLEARANCE between the pier and each bank: the pier is 7.5 cm wide
# (one block length), so the span must be at least 7.5 + 2·2 = 11.5 cm.
DEFAULT_BANK_A = np.array([0.45, -0.06, 0.03])
DEFAULT_BANK_B = np.array([0.45, 0.06, 0.03])

# False: build only the piers, no deck blocks on top.
BUILD_DECK = False

# Table height in the base frame. MEASURE.
TABLE_Z = 0.0

# Supply stacks: blocks lie flat, stacked straight up on the table, all with the
# same yaw. (x, y, yaw of the long side, number of blocks in the stack).
# Blocks are taken from the top of the first stack until it is empty, then from
# the next stack. TO DECIDE: must be reachable and out of the way of the bridge.
SUPPLY_STACKS = [
    (0.0125, 0.55, math.pi / 2, 4),
]

# How far below a block's top face the TCP (fingertip centre) goes to grasp it.
# Half a block grips it at mid-height. The finger pads reach below the TCP, so
# with a block underneath, keep this small enough that the pads stay above it.
# MEASURE with the real fingers: with thin blocks the pads may reach the block below.
GRASP_DEPTH = BLOCK_H / 2

MIN_BANK_BEARING = 0.02  # minimum length of deck resting on each bank
DECK_GAP = 0.004         # gap between adjacent deck ends, so placing the second one does not hit the first
FINGER_CLEARANCE = 0.02  # minimum gap wherever a finger has to go between two things
LAYER_GAP = 0.004        # minimum gap between the two blocks of a layer when no finger goes in it
HEIGHT_TOLERANCE = 0.005 # allowed height difference between pier top and bank top
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

    if BUILD_DECK:
        # Deck blocks are laid end to end:
        # (n_piers + 1)·L + n_piers·gap = span + 2·bank_bearing
        n_decks = math.ceil((span + 2 * MIN_BANK_BEARING + DECK_GAP) / (BLOCK_L + DECK_GAP))
        n_decks = max(n_decks, 1)
        n_piers = n_decks - 1
        bank_bearing = (n_decks * BLOCK_L + n_piers * DECK_GAP - span) / 2
        start = bank_a - u * bank_bearing  # where the first deck block starts (on bank A)
        start[2] = bank_z
        pier_centers = [
            start + u * ((k + 1) * BLOCK_L + (k + 0.5) * DECK_GAP) for k in range(n_piers)
        ]
        report.append(f"Span {span * 100:.1f} cm -> {n_decks} deck block(s), {n_piers} pier(s)")
        report.append(f"Deck rests {bank_bearing * 100:.1f} cm on each bank")
        if bank_bearing > BLOCK_L / 2:
            warnings.append("The deck rests more than half a block on the banks: the span is short; use fewer piers or move the banks closer")
    else:
        # No deck: one pier midway between the banks.
        middle = (bank_a + bank_b) / 2
        middle[2] = bank_z
        pier_centers = [middle]
        report.append(f"Span {span * 100:.1f} cm, no deck -> 1 pier midway between the banks")

    if pier_centers:
        # Gap between a bank's inner edge and the nearest pier's outer face.
        # One of the layers is gripped on the faces that point at the banks,
        # so a finger goes in there.
        bank_gap = abs(float(np.dot(pier_centers[0] - bank_a, u))) - pier_half
        report.append(f"Gap between each bank and the pier: {bank_gap * 100:.1f} cm")
        if bank_gap < FINGER_CLEARANCE - 1e-9:
            warnings.append(
                f"Bank-to-pier gap {bank_gap * 100:.1f} cm is less than the "
                f"{FINGER_CLEARANCE * 100:.1f} cm the fingers need: move the banks apart"
            )

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

    placements: list[Placement] = []
    for k, pier_center in enumerate(pier_centers):
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

    # The deck blocks go on last. With BUILD_DECK = False only the piers are built;
    # the banks still set where the piers go and how many layers they have.
    if BUILD_DECK:
        deck_z = pier_top + BLOCK_H / 2
        for i in range(n_decks):
            c = start + u * (i * (BLOCK_L + DECK_GAP) + BLOCK_L / 2)
            c[2] = deck_z
            placements.append(Placement(f"deck{i + 1}", c, wrap_yaw(axis_yaw)))
    else:
        report.append("BUILD_DECK = False: no deck blocks")

    n_supply = sum(stack[3] for stack in SUPPLY_STACKS)
    if len(placements) > n_supply:
        warnings.append(f"{len(placements)} blocks are needed but the supply stacks hold only {n_supply}")
    if GRASP_DEPTH > BLOCK_H:
        warnings.append("GRASP_DEPTH is more than a block high: the fingers would reach the block below")

    return placements, report, warnings, max(bank_z, pier_top)


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
    """Draws the banks, the supply stacks and the blocks in Swift, and moves the
    blocks with the gripper during the preview. Preview only: none of this is
    sent to the real robot."""

    BANK_COLOR = (0.55, 0.45, 0.35)
    SUPPLY_COLOR = (0.6, 0.6, 0.6)
    CARRIED_COLOR = (0.9, 0.6, 0.2)
    PLACED_COLOR = (0.2, 0.6, 0.9)
    BANK_LENGTH = 0.10  # how far each drawn bank reaches out from its inner edge
    BANK_WIDTH = 0.15   # drawn bank size across the bridge

    def __init__(self, panda, visualizer, bank_a, bank_b, placements):
        import spatialgeometry as sg

        self._sg = sg
        self.panda = panda
        self.visualizer = visualizer
        self.placements = placements
        self.blocks = []     # one per supply pick, in pick order
        self.n_grasped = 0
        self.carried = None  # (block, offset from the TCP) while held
        try:
            self._add_banks(bank_a, bank_b)
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

    def _add_banks(self, bank_a, bank_b):
        u = bank_b[:2] - bank_a[:2]
        u = u / np.linalg.norm(u)  # from bank A towards bank B
        yaw = math.atan2(u[1], u[0])
        for edge, outward in ((bank_a, -u), (bank_b, u)):
            height = edge[2] - TABLE_Z
            if height <= 0:
                continue
            centre_xy = edge[:2] + outward * self.BANK_LENGTH / 2
            pose = SE3.Trans(centre_xy[0], centre_xy[1], TABLE_Z + height / 2) * SE3.Rz(yaw)
            self._box([self.BANK_LENGTH, self.BANK_WIDTH, height], pose, self.BANK_COLOR)

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
        scene = BuildScene(panda, visualizer, bank_a, bank_b, placements)
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
