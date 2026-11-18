"""使用课程的 RobotWrapper 完成一次真实抓取并抬升。

这个文件只定义抓取行为；Swift 预演、确认提示和真机轨迹执行都由
RobotWrapper.run_joint_trajectory() 负责。

运行前请根据实验台实际位置修改 GRASP_POSITION。程序不会回 home，
不会创建虚拟物体，也不会在结束时自动松开夹爪。
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky.robot_wrapper import RobotWrapper


# Panda 基坐标系中的抓取中心 [x, y, z]，单位为米。
GRASP_POSITION = np.array([0.45, 0.15, 0.18])

# 从抓取点上方 10 cm 接近，抓住后再抬高 10 cm。
APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.10

# 夹爪参数：此处假设物体宽 2.5 cm。
OBJECT_WIDTH = 0.025
OPEN_WIDTH = 0.07
GRIPPER_SPEED = 0.05
GRASP_FORCE = 20.0
GRASP_TOLERANCE = 0.005

# 课程演示使用的低速轨迹参数。
RELATIVE_VELOCITY = 0.02
RELATIVE_ACCELERATION = 0.01
RELATIVE_JERK = 0.05

# False：只在 Swift 中预演（在家/没连实验台时用）；
# True：连接实验台 miniPC，每段预演后输入 yes 再让真机执行。
RUN_ON_REAL = False


def pose_at(start_pose: SE3, position: np.ndarray) -> SE3:
    """保持当前末端朝向，只改变末端位置。"""

    return SE3.Rt(start_pose.R, position)


def plan_segment(
    panda: rtb.Robot,
    generator: RuckigMotionGenerator,
    q_start: np.ndarray,
    target_pose: SE3,
) -> tuple[list[np.ndarray], float]:
    """规划一段平滑的笛卡尔运动，并转换为关节轨迹。"""

    start_pose = panda.fkine(q_start)
    cartesian_trajectory, dt = generator.calculate_cartesian_pose_trajectory(
        start_pose,
        target_pose,
        RELATIVE_VELOCITY,
        RELATIVE_ACCELERATION,
        RELATIVE_JERK,
    )
    q_trajectory = generator.cartesian_pose_to_joint_trajectory(
        panda,
        q_start,
        cartesian_trajectory,
    )
    return q_trajectory, dt



def main() -> None:
    panda = rtb.models.Panda()
    generator = RuckigMotionGenerator()

    # 课程推荐的高层接口：
    # has_visualizer=True -> 先在 Swift 中预演；
    # run_on_real=True    -> 接口询问 yes 后运行同一轨迹到真机。
    robot = RobotWrapper(
        has_visualizer=True,
        run_on_real=RUN_ON_REAL,
        panda_rtb_model=panda,
        q_start=panda.qr,
    )

    try:
        q_start = np.asarray(robot.get_current_joint_positions())
        start_pose = panda.fkine(q_start)

        approach_position = GRASP_POSITION + np.array(
            [0.0, 0.0, APPROACH_HEIGHT]
        )
        lift_position = GRASP_POSITION + np.array([0.0, 0.0, LIFT_HEIGHT])

        approach_pose = pose_at(start_pose, approach_position)
        grasp_pose = pose_at(start_pose, GRASP_POSITION)
        lift_pose = pose_at(start_pose, lift_position)

        # 先完成全部规划；此时真机还没有运动。
        q_approach, dt_approach = plan_segment(
            panda,
            generator,
            q_start,
            approach_pose,
        )
        q_grasp, dt_grasp = plan_segment(
            panda,
            generator,
            np.asarray(q_approach[-1]),
            grasp_pose,
        )
        q_lift, dt_lift = plan_segment(
            panda,
            generator,
            np.asarray(q_grasp[-1]),
            lift_pose,
        )

        print("Grasp target:", GRASP_POSITION, "m")
        print("1. Move above the grasp point")
        if not robot.run_joint_trajectory(q_approach, dt_approach):
            return

        # RobotWrapper 只封装机械臂轨迹；夹爪仍通过它持有的课程真机接口调用。
        # run_on_real=False 时 robot.real_robot 是 None，只能操作 Swift 里的夹爪。
        robot.visualizer.move_gripper(OPEN_WIDTH, GRIPPER_SPEED)
        if robot.real_robot is not None:
            robot.real_robot.gripper.open(GRIPPER_SPEED)

        print("2. Descend vertically to the grasp pose")
        if not robot.run_joint_trajectory(q_grasp, dt_grasp):
            return

        print("3. Close the gripper and check the grasp")
        robot.visualizer.move_gripper(OBJECT_WIDTH, GRIPPER_SPEED)
        if robot.real_robot is None:
            grasped = True  # 纯仿真没有真实物体，默认抓取成功
        else:
            grasped = robot.real_robot.gripper.grasp(
                width=OBJECT_WIDTH,
                speed=GRIPPER_SPEED,
                force=GRASP_FORCE,
                epsilon_inner=GRASP_TOLERANCE,
                epsilon_outer=GRASP_TOLERANCE,
            )
        if not grasped:
            robot.real_robot.gripper.stop()
            print("The gripper did not confirm a grasp; cancelling the lift.")
            return

        print("4. Grasp succeeded, lifting vertically")
        if not robot.run_joint_trajectory(q_lift, dt_lift):
            print("Lift cancelled by the user; the object is still held.")
            return

        print("Grasp complete; the object is held in the gripper.")
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
