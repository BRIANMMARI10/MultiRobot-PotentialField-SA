import sys
sys.path.append('./src')
sys.stdout = open("simulation_output_sa_peturb.txt", "w")

import csv
import os
import math
import random
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Suppress GUI plots
import matplotlib.pyplot as plt

from Simulation.ComApi import *
from Simulation.ComSurfacePotentialField import ComSurfacePotentialField
from Simulation.ComObjectCollection import updateObstacle_kdtree
import Simulation.ComObjectCollection as ComCol

# Results CSV setup
csv_filename = 'simulation_results_sa_peturb.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Simulation', 'Robot ID', 'Steps Taken', 'Goal Reached', 'Path Length', 'Time Taken', 'Collisions'])

# Simulation parameters
N_values = [1, 2, 4, 8, 16]
num_trials = 3

results = {
    "N": [],
    "avg_path_length": [],
    "avg_time": [],
    "total_collisions": []
}

# Main simulation loop
for N in N_values:
    for trial in range(num_trials):
        print(f"\n--- Running N={N}, Trial={trial} ---")
        ComCol.clearAll()

        # Reset class-level state in ComRobot
        ComRobot._robot_count = 0
        ComRobot._robot_list = []

        # Move *everything* stage-related inside the trial loop
        stage = ComStage2D()
        stage.enableFigSave('../results/planning-2D')
        stage.enablePosSave('../results/planning-2D')
        stage.mRuningTime = 30
        stage.isPlotGraph = True
        stage.isShowCommunicated = False
        stage.setEnvSize((1000, 1000, 1000))
        stage.setFigSize((8, 8))
        stage.addObstacleType('ComFish')

        total_path_length = 0
        total_time = 0
        total_collisions = 0
        total_robots = 0

        # Now create robots with optimal start position
        robot_list = []
        for _ in range(N):
            start_x = 800 + random.uniform(-50, 50)
            start_y = -800 + random.uniform(-50, 50)
            robot = ComRobotCon((start_x, start_y, 0))
            robot.setColor((0, 0.5, 0.5, 1))
            robot.mRobotType = '2D'
            robot.mCommunicationRange = 800
            robot.isShowSenseRange = False
            robot.isPlotOrientationLine = True
            robot.isPlotTrail = True
            robot.setPlanningTarget((-800, 800, 0))

            base_theta = random.uniform(0, 2 * math.pi)
            # Apply perturbation: ±10 degrees in radians
            perturb = True  # Toggle this to False for baseline runs
            epsilon = random.uniform(-10, 10) * (math.pi / 180) if perturb else 0
            robot.setDirection(base_theta + epsilon)
            robot.isPathPlanning = True
            robot.getPlanningControl().setStride(15)
            robot.setRadius(15)
            robot.setTrailLineColor((1, 0, 1, 0.5))
            robot.setShape('circle')
            stage.addRobot(robot)
            robot_list.append(robot)

        # Add obstacles
        obstacles_pos = [
            (random.uniform(-500, 500), random.uniform(-500, 500), 0)
            for _ in range(10)
        ]
        for pos in obstacles_pos:
            fish = ComFish(pos)
            fish.isPlotTargetLine = False
            fish.mSpeed = 0
            fish.setShape('square')
            fish.setColor((0.3, 0.3, 0.3, 0.5))
            fish.setRadius(40)
            stage.addStuff(fish)

        updateObstacle_kdtree(['ComFish'])
        fish = ComCol.getObjectByType('ComFish')

        # Temporary potential field for initial robot positioning only (no population needed here)
        temp_field = ComSurfacePotentialField()
        temp_field.setZDir('2D')
        temp_field.setX(np.arange(-1000, 1000, 10))
        temp_field.setY(np.arange(-1000, 1000, 10))
        temp_field.setObstacleList(fish)
        temp_field.setTarget((-800, 800))
        temp_field.setPopulation([])  # Explicitly set an empty list here
        temp_field.update()

        # Calibrate repulsion weight using simulated annealing
        temp_field.calibrate_repulsion_weight()

        # Use the tuned repulsion weight in the main potential field
        optimal_repulsion_weight = temp_field.repulsion_weight

        # Now set up main potential field (with robots)
        potential_field = ComSurfacePotentialField()
        potential_field.setRepulsionWeight(optimal_repulsion_weight)
        potential_field.setZDir('2D')
        potential_field.setX(np.arange(-1000, 1000, 10))
        potential_field.setY(np.arange(-1000, 1000, 10))
        potential_field.setObstacleList(fish)
        potential_field.setPopulation(robot_list)  # robots created, now assign
        potential_field.setTarget((-800, 800))
        potential_field.update()

        # Add potential field visualization
        stage.addSurface(potential_field)

        stage.run()

        for robot in robot_list:
            print(f"Trial={trial}, Robot={robot.mId}, PathLogLen={len(robot.robot_path_log) if hasattr(robot, 'robot_path_log') else 'None'}")

            if hasattr(robot, "robot_path_log") and len(robot.robot_path_log) > 1:
                steps_taken = len(robot.robot_path_log)
                final_pos = robot.robot_path_log[-1]
                goal = np.array([-800, 800])
                distance_to_goal = robot.distance(final_pos, goal)
                goal_reached = distance_to_goal <= 20
                path_length = sum(
                    robot.distance(robot.robot_path_log[i], robot.robot_path_log[i+1])
                    for i in range(len(robot.robot_path_log) - 1)
                )
            else:
                steps_taken = 0
                goal_reached = False
                path_length = 0

            time_taken = robot.goal_reached_time if robot.goal_reached_time is not None else stage.mRuningTime
            collisions = getattr(robot, 'collisions', 0)

            total_path_length += path_length
            total_time += time_taken
            total_collisions += collisions
            total_robots += 1

            # After stage.run() and metrics computation:
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    f"N={N}, Trial={trial}",
                    robot.mId,
                    steps_taken,
                    goal_reached,
                    round(path_length, 2),
                    round(time_taken, 2),
                    collisions
                ])
        stage.clear()

import pandas as pd

# Read the CSV you just wrote
df = pd.read_csv(csv_filename)

# Create directory for histograms
os.makedirs("results/histograms", exist_ok=True)

# Metrics to plot
metrics = ['Steps Taken', 'Path Length', 'Time Taken', 'Collisions']

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.hist(df[metric], bins=20, edgecolor='black')
    plt.title(f'Histogram of {metric}')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/histograms/{metric.replace(" ", "_").lower()}_histogram.png')
    plt.close()
