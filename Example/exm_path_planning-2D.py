import sys
sys.path.append('./src')
sys.stdout = open("simulation_output.txt", "w")
import csv
import os
# Create or overwrite the results file
csv_filename = 'simulation_results.csv'
# Only create file and write header if it doesn't exist
# Always clear and write header at the start
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Simulation', 'Robot ID', 'Steps Taken', 'Goal Reached', 'Path Length', 'Time Taken', 'Collisions'])

from Simulation.ComApi import *
import random
import math
from Simulation.ComSurfacePotentialField import ComSurfacePotentialField
from Simulation.ComObjectCollection import updateObstacle_kdtree
import Simulation.ComObjectCollection as ComCol
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <- Use non-GUI backend to suppress all plot windows
import matplotlib.pyplot as plt


N_values = [1, 2, 4, 8, 16]
num_trials = 3

results = {
    "N": [],
    "avg_path_length": [],
    "avg_time": [],
    "total_collisions": []
}

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
        stage.isPlotGraph = False
        stage.isShowCommunicated = False
        stage.setEnvSize((1000, 1000, 1000))
        stage.setFigSize((8, 8))
        stage.addObstacleType('ComFish')

        total_path_length = 0
        total_time = 0
        total_collisions = 0
        total_robots = 0

        robot_list = []
        for i in range(N):
            # range_x = (-stage.mEnvSize[0], stage.mEnvSize[0])
            # range_y = (-stage.mEnvSize[1], stage.mEnvSize[1])
            # range_z = (-stage.mEnvSize[2], stage.mEnvSize[2])
            # x = random.uniform(range_x[0], range_x[1])
            # y = random.uniform(range_y[0], range_y[1])
            # z = random.uniform(range_z[0], range_z[1])
            # robot = ComRobotCon((800, -800, 0))
            # Brian
            start_x = 800 + random.uniform(-50, 50)
            start_y = -800 + random.uniform(-50, 50)
            robot = ComRobotCon((start_x, start_y, 0))
            # Brian

            robot.setColor((0, 0.5, 0.5, 1))
            robot.mRobotType = '2D'
            robot.mCommunicationRange = 800
            robot.isShowSenseRange = False
            robot.isDrawCommunicationRange = False 
            robot.isPlotTargetLine = False
            robot.isPlotOrientationLine = True
            robot.isPlotTrail = True
            robot.setPlanningTarget((-800, 800, 0))
            # robot.setTargetDirection(0)
            robot.setDirection(random.uniform(0, 2 * math.pi))

            robot.isPathPlanning = True
            robot.getPlanningControl().setStride(15)
            print(robot.getPlanningControl())
            robot.setRadius(15)
            robot.setTrailLineColor((1, 0, 1, 0.5))

            # Brian
            robot.setShape('circle')
            # Brian
            stage.addRobot(robot)
            # Brian
            robot_list.append(robot)  # <--- save it
            # Brian
        
        # Brian
        obstacles_pos = [
            (
            random.uniform(-500, 500), 
            random.uniform(-500, 500), 
            0
            )
            for _ in range(10)  # You can increase number of obstacles here
        ]
        # Brian
        for obj in obstacles_pos:
            x = obj[0]
            y = obj[1]
            z = obj[2]
            fish = ComFish((x, y, z))
            fish.isPlotTargetLine = False
            fish.isPlotTrail = False
            fish.isShowSenseRange = True
            fish.mSpeed = 0
            fish.setShape('square')
            fish.setColor((0.3, 0.3, 0.3, 0.5))
            fish.setRadius(40)
            stage.addStuff(fish)
    
        # Brian
        updateObstacle_kdtree(['ComFish'])  # Build KDTree before simulation
        # Brian

        food = ComCol.getObjectByType('ComFish')
        population = ComCol.getObjectByType('ComRobot')

        CommunicationDist = 800
        surfz = ComSurfacePotentialField()
        surfz.setZDir('2D')
        surfz.setX(np.arange(-1000, 1000, 10))
        surfz.setY(np.arange(-1000, 1000, 10))
        surfz.setAlpha(1.0)
        surfz.setObstacleList(food)
        surfz.setPopulation(population)
        surfz.isShowAgentMark = True
        surfz.setRobotRadius(75)
        # surfz.setCMap(plt.cm.hot)
        stage.addSurface(surfz)

        stage.mElapsedTime = 0  # <-- Reset elapsed time before starting the trial
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
