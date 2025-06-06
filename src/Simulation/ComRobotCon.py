from Simulation.ComRobot import ComRobot 
from Simulation.ComObject import ComObject 
from Simulation.ComObjectCollection import getNearestObstacle  # Make sure this is imported
import copy
import math
isCupy = False
try:
    import cupy as np
    isCupy = True
except:
    import numpy as np
    isCupy = False
from Common import settings
from Common.DrKDtree import KDtree
import Common.settings as mySettings 
import random
from Common import utils
from Simulation.ComPathPlanning import ComPathPlanning
from Simulation.ComPathPlanning3D import ComPathPlanning3D
import Simulation.ComObjectCollection as ComCol
collision_threshold = 55 


class PathFinderController:
    """
    Constructs an instantiate of the PathFinderController for navigating a
    3-DOF wheeled robot on a 2D plane

    Parameters
    ----------
    Kp_rho : The linear velocity gain to translate the robot along a line
             towards the goal
    Kp_alpha : The angular velocity gain to rotate the robot towards the goal
    Kp_beta : The offset angular velocity gain accounting for smooth merging to
              the goal angle (i.e., it helps the robot heading to be parallel
              to the target angle.)
    """

    def __init__(self, Kp_rho, Kp_alpha, Kp_beta):
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def calc_control_command(self, x_diff, y_diff, theta, theta_goal):
        """
        Returns the control command for the linear and angular velocities as
        well as the distance to goal

        Parameters
        ----------
        x_diff : The position of target with respect to current robot position
                 in x direction
        y_diff : The position of target with respect to current robot position
                 in y direction
        theta : The current heading angle of robot with respect to x axis
        theta_goal: The target angle of robot with respect to x axis

        Returns
        -------
        rho : The distance between the robot and the goal position
        v : Command linear velocity
        w : Command angular velocity
        """

        # Description of local variables:
        # - alpha is the angle to the goal relative to the heading of the robot
        # - beta is the angle between the robot's position and the goal
        #   position plus the goal angle
        # - Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards
        #   the goal
        # - Kp_beta*beta rotates the line so that it is parallel to the goal
        #   angle
        #
        # Note:
        # we restrict alpha and beta (angle differences) to the range
        # [-pi, pi] to prevent unstable behavior e.g. difference going
        # from 0 rad to 2*pi rad with slight turn

        rho = np.hypot(x_diff, y_diff)
        alpha = (np.arctan2(y_diff, x_diff)
                 - theta + np.pi) % (2 * np.pi) - np.pi
        beta = (theta_goal - theta - alpha + np.pi) % (2 * np.pi) - np.pi
        v = self.Kp_rho * rho
        w = self.Kp_alpha * alpha - self.Kp_beta * beta

        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v

        return rho, v, w

class ComRobotCon(ComRobot):
    def __init__(self, pos):
        super(ComRobotCon, self).__init__(pos)
        self.mPathPlanningControl = None  # <- leave this as None
        self.mPathPlanningControl3D = None  # <- same for 3D
        self.mTargetLineLen = 300.0       # 目标线的长度
        self.mObjectType = "ComRobot"       # 用于标识当前物体类别
        self.isPlotOrientationLine = True
        self.mOrientationLineLen = 200.00
        self.mLineSpeed = 0.0               # 水平方向速度
        self.mZLineSpeed = 0.0              # 垂直方向速度
        self.mRotationSpeed = 0.0
        self.isPathPlanning = False
        self.mPlanningTarget = None
        self.mMaxLinearSpeed = 150 
        self.mMaxZLinearSpeed = 100
        self.mMaxAngularSpeed = 7
        self.mPathFollowingControl = PathFinderController(
            mySettings.PATH_FOLLOWING_K_RHO,
            mySettings.PATH_FOLLOWING_K_ALPHA,
            mySettings.PATH_FOLLOWING_K_BETA 
            )
        # self.mPathFollowingControl = PathFinderController(
        #     9,
        #     15,
        #     3
        #     )
        self.mPathPlanningControl = ComPathPlanning()
        self.mPathPlanningControl3D = ComPathPlanning3D()
        # self.mOrientationLineColor = 'blue'
        self.robot_path_log = []  # Brian: stores the real path during simulation
        self.goal_reached_time = None # Brian: time when robot reaches goal
        self.already_colliding = False # Brian: not yet collided
    
    def getPlanningControl(self):
        if self.mRobotType == '2D':
            return self.mPathPlanningControl
        elif self.mRobotType == '3D':
            return self.mPathPlanningControl3D

    # Brian
    def setPlanningTarget(self, pos):
        if pos is not None and isinstance(pos, (list, tuple, np.ndarray)):
            self.getPlanningControl().setTarget(pos)
        else:
            # Fallback target
            self.getPlanningControl().setTarget(np.array([-800, 800, 0]))
    # Brian

    def move(self):
        """
        向目标移动一步
        :return:
        """
        print(f"[Move] Robot {self.mId}: Iteration = {self.mIterations}")
        self.mIterations += 1
        pos_last = self.mPos.copy()

        # Brian: record position BEFORE moving
        self.robot_path_log.append(self.getPos2d().copy())
        # Brian

        # Brian: check if robot has reached goal (only once)
        if self.goal_reached_time is None:
            goal = np.array([-800, 800])
            distance_to_goal = self.distance(self.getPos2d(), goal)
            if distance_to_goal <= 20:  # Same as goal_threshold
                self.goal_reached_time = self.mIterations * mySettings.TIME_INTERVAL

        robot_pos = self.pos
        target_pos = self.target

        if self.mRobotType == '2D':
            # 机器人旋转
            direction_next = self.mRotationSpeed * mySettings.TIME_INTERVAL + self.mDirection
            self.setDirection(direction_next)
            pos_next = np.array([self.mLineSpeed * mySettings.TIME_INTERVAL, 0])
            rot_mat = self.getRotationMat(self.mDirection)
            pos_next = np.matmul(pos_next, rot_mat)
            pos_next += pos_last[0:2]
            self.pos = pos_next

            # Brian: Collision check
            nearest_obstacle = getNearestObstacle(self.getPos2d())
            if nearest_obstacle is not None:
                nearest_obstacle = np.squeeze(nearest_obstacle)[:2]
                dist_to_nearest = self.distance(self.getPos2d(), nearest_obstacle)
                collision_now = dist_to_nearest <= collision_threshold
            
                if collision_now and not hasattr(self, 'already_colliding'):
                    self.already_colliding = False
                if collision_now and not self.already_colliding:
                    if not hasattr(self, 'collisions'):
                        self.collisions = 0

                    if collision_now and not self.already_colliding:
                        self.collisions += 1
                        self.already_colliding = True
                    elif not collision_now:
                        self.already_colliding = False
            # Brian end
        elif self.mRobotType == '3D':
            direction_next = self.mRotationSpeed * mySettings.TIME_INTERVAL + self.mDirection
            self.setDirection(direction_next)
            pos_next = np.array((0.,0.,0.))
            pos_diff_xy = np.array([self.mLineSpeed * mySettings.TIME_INTERVAL, 0])
            pos_diff_z = self.mZLineSpeed * mySettings.TIME_INTERVAL
            rot_mat = self.getRotationMat(self.mDirection)
            pos_diff_xy = np.matmul(pos_diff_xy, rot_mat)
            pos_next[0:2] = pos_diff_xy
            pos_next[2] = pos_diff_z
            pos_next += pos_last
            self.pos = pos_next

        self.setShape(shape=self.mShape)
        if not (self.mPos == pos_last).all():
            self.mTrail[0].append(self.mPos[0])
            self.mTrail[1].append(self.mPos[1])
            self.mTrail[2].append(self.mPos[2])
    
    def draw(self, ax):
        super().draw(ax)
        

        if self.mStage.mStageType == '2D':
            if self.isPathPlanning:
                if self.getPlanningControl().mPathPtList_x is not None:
                    ax.plot(self.getPlanningControl().mPathPtList_x, self.getPlanningControl().mPathPtList_y, 'k--')
            if self.isPlotOrientationLine:
                rot_mat = self.getRotationMat(self.mDirection)
                # 旋转
                target = np.array([self.mOrientationLineLen, 0])
                target = np.matmul(target, rot_mat)
                # 平移
                target = np.array([self.mPos[0]+target[0], self.mPos[1]+target[1]])
                # print(target)
                x = np.array([self.mPos[0], target[0]])
                y = np.array([self.mPos[1], target[1]])
                if isCupy:
                    x = x.get()
                    y = y.get()
                
                ax.plot(x, y, 'b--')
        elif self.mStage.mStageType == '3D':
            if self.isPathPlanning:
                if self.getPlanningControl().mPathPtList_x is not None:
                    ax.plot(self.getPlanningControl().mPathPtList_x, self.getPlanningControl().mPathPtList_y, self.getPlanningControl().mPathPtList_z, 'k--')
            if self.isPlotOrientationLine:
                rot_mat = self.getRotationMat(self.mDirection)
                # 旋转
                target = np.array([self.mOrientationLineLen, 0])
                target = np.matmul(target, rot_mat)
                # 平移
                target = np.array([self.mPos[0]+target[0], self.mPos[1]+target[1]])
                # print(target)
                print(f"[Planning] Robot {self.mId}: New Target = {self.mTarget}")
                x = np.array([self.mPos[0], target[0]])
                y = np.array([self.mPos[1], target[1]])
                z = np.array([self.mPos[2], self.mPos[2]])
                if isCupy:
                    x = x.get()
                    y = y.get()
                
                ax.plot(x, y, z, 'b--')


    def update(self):
        self.sense()
        self.processInfo()

        # if self.isPathPlanning and self.isClosedToTarget():
        if self.isPathPlanning:
            if self.getPlanningControl().mTarget is None:
                self.setPlanningTarget(self.mPos)
            self.pathPlanning()

            # print(self.mDirection)
            # print(self.mTargetDirection)
            # print(self.mPathPlanningControl.mPathPtList_x, self.mPathPlanningControl.mPathPtList_y)
        self.pathFollowing()
        # if self.mId == 0:
        #     print(self.mLineSpeed, self.mRotationSpeed)
        self.move()
        print(f"[Update] Robot {self.mId}: Pos = {self.mPos}, Target = {self.mTarget}, Speed = {self.mLineSpeed}, Rotation = {self.mRotationSpeed}")

    def setDirection(self, direction):
        if direction > 2 * math.pi:
            direction %= (2*math.pi) 
        if direction < -2 * math.pi:
            direction %= (-2 * math.pi)
        # if direction < 0:
        #     direction += 2 * math.pi
            
        super().setDirection(direction)

    def pathPlanning(self):
        if self.mRobotType == '2D':
            self.getPlanningControl().setRobotRadius(75)
            self.getPlanningControl().setPos(self.mPos)
            self.getPlanningControl().update()
            self.getPlanningControl().setEnvSize(self.mStage.mEnvSize)
            x, y, angle = self.getPlanningControl().getNextDest()

            if x is None or y is None:
                self.setTarget(self.mPos)
            else:
                self.setTarget((x, y, 0))
            if angle is not None:
                self.setTargetDirection(angle)
        elif self.mRobotType == '3D':
            self.getPlanningControl().setRobotRadius(75)
            self.getPlanningControl().setPos(self.mPos)
            self.getPlanningControl().update()
            self.getPlanningControl().setEnvSize(self.mStage.mEnvSize)
            x, y, z, angle = self.getPlanningControl().getNextDest()

            if x is None or y is None:
                self.setTarget(self.mPos)
            else:
                self.setTarget((x, y, z))
            if angle is not None:
                self.setTargetDirection(angle)

    def pathFollowing(self):
        '''
        路径跟随，根据当前机器人的位置和方向，以及目标位置和方向，修改机器人的线速度和角速度
        '''
        if self.mRobotType == '2D':
            x_diff = self.mTarget[0] - self.mPos[0]
            y_diff = self.mTarget[1] - self.mPos[1]
            theta = self.mDirection
            theta_goal = self.mTargetDirection
            rho, v, w = self.mPathFollowingControl.calc_control_command(
            x_diff, y_diff, theta, theta_goal)
            if abs(v) > self.mMaxLinearSpeed:
                v = np.sign(v) * self.mMaxLinearSpeed

            if abs(w) > self.mMaxAngularSpeed:
                w = np.sign(w) * self.mMaxAngularSpeed

            self.mLineSpeed = v
            self.mRotationSpeed = w
        elif self.mRobotType == '3D':
            x_diff = self.mTarget[0] - self.mPos[0]
            y_diff = self.mTarget[1] - self.mPos[1]
            z_diff = self.mTarget[2] - self.mPos[2]
            theta = self.mDirection
            if self.mTargetDirection is None:
                theta_goal = self.mDirection
            else:
                theta_goal = self.mTargetDirection
            rho, v, w = self.mPathFollowingControl.calc_control_command(
            x_diff, y_diff, theta, theta_goal)
            z_dist = z_diff
            z_v = mySettings.PATH_FOLLOWING_K_RHO * z_dist
            if abs(v) > self.mMaxLinearSpeed:
                v = np.sign(v) * self.mMaxLinearSpeed

            if abs(w) > self.mMaxAngularSpeed:
                w = np.sign(w) * self.mMaxAngularSpeed
            
            if abs(z_v) > self.mMaxZLinearSpeed:
                z_v = np.sign(z_v) * self.mMaxZLinearSpeed

            self.mLineSpeed = v
            self.mZLineSpeed = z_v
            self.mRotationSpeed = w
            print(f"[Control] Robot {self.mId}: v = {self.mLineSpeed}, w = {self.mRotationSpeed}")
            



    def stop(self):
        self.mLineSpeed = 0
        self.mRotationSpeed = 0

    def isClosedToTarget(self):
        if self.distance(self.mPos, self.mTarget) < 5:
            return True
        else:
            return False

    def isStopping(self):
        if self.mLineSpeed < 0.00001 and self.mRotationSpeed < 0.00001:
            return True
        else:
            return False
    