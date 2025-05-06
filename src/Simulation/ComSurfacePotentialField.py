from matplotlib import cm, markers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from Common import utils as myUtils
isCupy = False
try:
    import cupy as np
    isCupy = True
except:
    import numpy as np
    isCupy = False
from Simulation.ComSurfaceBase import ComSurfaceBase, PlotType
from Simulation.ComPathPlanning import calc_potential_field2
from Simulation.ComPathPlanning3D import potential_field_planning, calc_potential_field_mat


class ComSurfacePotentialField(ComSurfaceBase):
    def __init__(self, ax=None) -> None:
        super().__init__(ax=ax)
        self.mObstacleList = []
        self.mTarget = None
        self.mCMap = cm.Blues
        self.mRobotRadius = 20      
        self.mPlotType = PlotType.type_contourf
        self.mBindingRobot = None
        # Brian
        self.repulsion_weight = 1.0  # Default repulsion multiplier for obstacle avoidance
        # Brian
        # self.mCMap = cm.ocean

    # Brian
    def setRepulsionWeight(self, weight: float):
        """Sets the repulsion weight for obstacle avoidance."""
        self.repulsion_weight = weight
    # Brian

    def setBindingRobot(self, robot):
        """Sets the binding robot for this object.

        Args:
            robot (object): The robot object to bind.
        """
        self.mBindingRobot = robot
        
    def setRobotRadius(self, radius):
        """Sets the radius of this object's associated robot.

        Args:
            radius (float): The radius to set for the robot.
        """
        self.mRobotRadius = radius
        
    def setTarget(self, target: tuple):
        """Sets the target location for this object's associated robot.

        Args:
            target (tuple): A tuple representing the x and y coordinates of the target location.
        """
        self.mTarget = target
        
    def setObstacleList(self, obstacle_list: list):
        """Sets the list of obstacle locations for this object's associated robot.

        Args:
            obstacle_list (list): A list of tuples representing the x and y coordinates of each obstacle.
        """
        self.mObstacleList = obstacle_list


    def update(self):
        """Updates the potential field based on current parameters."""

        """
        人工势场计算 (Translation: Artificial potential field calculation)
        """

        # Create a list of obstacle positions
        obstacle_pos_group = [obstacle.mPos for obstacle in self.mObstacleList]

        # If there is a bound robot, set its offset
        if self.mBindingRobot is not None:
            self.setOffset(self.mBindingRobot.mPos[2])

        # If the Z direction is 'z' or '2D'
        if self.mZDir == 'z' or self.mZDir == '2D':

            # If there are obstacles
            if len(self.mObstacleList) > 0:

                # Create lists of X and Y coordinates from each obstacle position
                obstacle_pos_x_list = [i[0] for i in obstacle_pos_group]
                obstacle_pos_y_list = [i[1] for i in obstacle_pos_group]

                # Calculate the potential field
                gx, gy = None, None
                ox = obstacle_pos_x_list
                oy = obstacle_pos_y_list
                reso= (np.max(self.mX) - np.min(self.mX))/len(self.mX)
                rr = self.mRobotRadius
                map_size=(np.min(self.mX), np.min(self.mY), np.max(self.mX), np.max(self.mY))

                # If there is a target, set the target coordinates
                if self.mTarget is not None:
                    gx, gy = self.mTarget[0:2]

                # Calculate the potential field data and store it in mData
                # Brian
                data, _, _ = calc_potential_field2(gx, gy, ox, oy, rr, reso, map_size, repulsion_weight=self.repulsion_weight)
                # Brian
                self.mData = np.array(data).T

        # If the Z direction is '3D'
        elif self.mZDir == '3D':

            # If there are obstacles
            if len(self.mObstacleList) > 0:

                # Create lists of X, Y, and Z coordinates from each obstacle position
                obstacle_pos_x_list = [i[0] for i in obstacle_pos_group]
                obstacle_pos_y_list = [i[1] for i in obstacle_pos_group]
                obstacle_pos_z_list = [i[2] for i in obstacle_pos_group]

                gx, gy, gz = None, None, None
                ox = obstacle_pos_x_list
                oy = obstacle_pos_y_list
                oz = obstacle_pos_z_list
                reso= (np.max(self.mX) - np.min(self.mX))/len(self.mX)
                rr = self.mRobotRadius
                map_size=(np.min(self.mX), np.min(self.mY), np.max(self.mX), np.max(self.mY))

                # If there is a target, set the target coordinates
                if self.mTarget is not None:
                    gx, gy, gz = self.mTarget[:]

                # Calculate the potential field data and store it in mData
                data, minx, miny, minz = calc_potential_field_mat(gx, gy, gz, ox, oy, oz, rr, reso=10, map_size=(-1000, -1000, -1000, 1000, 1000, 1000))
                offset_z = int((self.mOffset-minz)/reso)
                self.mData = data[:,:,offset_z]

        # If the Z direction is 'y'
        elif self.mZDir == 'y':
            pass 

        # If the Z direction is 'x'
        elif self.mZDir == 'x':
            pass 

        # Call the update method of the parent class
        super().update()


    def draw(self):
        """Draws the object."""
        super().draw()
    
    # Brian
    def calibrate_repulsion_weight(self, initial_weight=1.0, iterations=50, initial_temp=50.0, cooling_rate=0.98):
        """
        Uses simulated annealing to find the best repulsion weight for obstacle avoidance.
        """
        self.repulsion_weight = initial_weight
        best_weight = initial_weight
        best_score = self.evaluate_field_quality()

        current_temp = initial_temp

        for _ in range(iterations):
            candidate_weight = self.repulsion_weight + np.random.uniform(-1, 1)
            candidate_weight = max(0.01, candidate_weight)  # Ensure it's positive

            threshold = np.random.uniform(5, 20)  # explore different threshold values
            self.repulsion_weight = candidate_weight
            self.update()  # Recompute field with new weight

            score = self.evaluate_field_quality(threshold=threshold)

            delta = score - best_score
            if delta < 0 or np.exp(-delta / current_temp) > np.random.rand():
                best_score = score
                best_weight = candidate_weight

            current_temp *= cooling_rate

        self.repulsion_weight = best_weight
        self.update()
        print(f"Optimal repulsion weight: {best_weight}")

    def evaluate_field_quality(self, threshold=5):
        """
        Evaluate the potential field quality.
        Combines mean potential and high-risk region penalties.
        Lower is better.

        Args:
            threshold (float): The potential value above which areas are considered risky.
        """
        if self.mData is None:
            return float("inf")

        # Smoothness component
        mean_potential = np.mean(self.mData)

        # Collision risk penalty: high repulsion zones
        high_potential_penalty = np.sum(self.mData > threshold)

        # You can adjust the weight multiplier (currently 10) - tune higher for more avoidance
        return mean_potential + 50 * high_potential_penalty
    # Brian
