#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json
import math
from random import uniform
import copy

import scipy
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point, PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import scipy.stats
from numpy.random import choice

np.set_printoptions(linewidth=200)

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def angle_to_neg_pi_to_pi(angle: float) -> float:
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


# see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
def ray_line_intersection(ray_origin, ray_direction_rad, point1, point2):
    # Convert to numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array([math.cos(ray_direction_rad), math.sin(ray_direction_rad)])
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)

    # Ray-Line Segment Intersection Test in 2D
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denominator = np.dot(v2, v3)
    if denominator == 0:
        return None
    t1 = np.cross(v2, v1) / denominator
    t2 = np.dot(v1, v3) / denominator
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return [ray_origin + t1 * ray_direction]
    return None


class Map:
    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb

    @property
    def top_right(self) -> Tuple[float, float]:
        return self.map_aabb[1], self.map_aabb[3]

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return self.map_aabb[0], self.map_aabb[2]

    def draw_distances(self, origins: List[Tuple[float, float]]):
        """Example usage:
        map_ = Map(obstacles, map_aabb)
        map_.draw_distances([(0.0, 0.0), (3, 3), (1.5, 1.5)])
        """

        # Draw scene
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        x_min_global, x_max_global, y_min_global, y_max_global = self.map_aabb
        for aabb in self.obstacles:
            width = aabb[1] - aabb[0]
            height = aabb[3] - aabb[2]
            rect = patches.Rectangle(
                (aabb[0], aabb[2]), width, height, linewidth=2, edgecolor="r", facecolor="r", alpha=0.4
            )
            ax.add_patch(rect)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Plot of Obstacles")
        ax.set_aspect("equal", "box")
        plt.grid(True)

        # Draw rays
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        for origin in origins:
            for angle in angles:
                closest_distance = self.closest_distance(origin, angle)
                if closest_distance is not None:
                    x = origin[0] + closest_distance * math.cos(angle)
                    y = origin[1] + closest_distance * math.sin(angle)
                    plt.plot([origin[0], x], [origin[1], y], "b-")
        plt.show()

    def closest_distance(self, origin: Tuple[float, float], angle: float) -> Optional[float]:
        """Returns the closest distance to an obstacle from the given origin in the given direction `angle`. If no
        intersection is found, returns `None`.
        """

        def lines_from_obstacle(obstacle: OBS_TYPE):
            """Returns the four lines of the given AABB format obstacle.
            Example usage: `point0, point1 = lines_from_obstacle(self.obstacles[0])`
            """
            x_min, x_max, y_min, y_max = obstacle
            return [
                [(x_min, y_min), (x_max, y_min)],
                [(x_max, y_min), (x_max, y_max)],
                [(x_max, y_max), (x_min, y_max)],
                [(x_min, y_max), (x_min, y_min)],
            ]

        # Iterate over the obstacles in the map to find the closest distance (if there is one). Remember that the
        # obstacles are represented as a list of AABBs (Axis-Aligned Bounding Boxes) with the format
        # (x_min, x_max, y_min, y_max).
        result = None
        origin = np.array(origin)

        for obstacle in self.obstacles:
            for line in lines_from_obstacle(obstacle):
                p = ray_line_intersection(origin, angle, line[0], line[1])
                if p is None:
                    continue

                dist = np.linalg.norm(np.array(p) - origin)
                if result is None:
                    result = dist
                else:
                    result = min(result, dist)
        return result

# PID controller class
######### Your code starts here #########
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp #giving intial to all gains
        self.ki = ki
        self.kd = kd
        self.prev_error = None
        self.integral = 0.0 #we want to skip the derivative first

    def control(self, error: float, dt: float) -> float:
        if dt <= 0: #we don't want to divide by zero
            dt = 1e-6
        self.integral += error * dt #we are summing all past errors
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt #getting change in error
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative #gain equation
        self.prev_error = error #update error
        return control_signal
    
    def reset(self):
        self.prev_error = None
        self.integral = 0.0
######### Your code ends here #########


class Particle:
    def __init__(self, x: float, y: float, theta: float, log_p: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_p = log_p

    def __str__(self) -> str:
        return f"Particle<pose: {self.x, self.y, self.theta}, log_p: {self.log_p}>"


class ParticleFilter:

    def __init__(
        self,
        map_: Map,
        n_particles: int,
        translation_variance: float,
        rotation_variance: float,
        measurement_variance: float,
    ):
        self.particles_visualization_pub = rospy.Publisher("/pf_particles", PoseArray, queue_size=10)
        self.estimate_visualization_pub = rospy.Publisher("/pf_estimate", PoseStamped, queue_size=10)

        # Initialize uniformly-distributed particles
        ######### Your code starts here #########
        self._particles = [] #get particle objects
        self._map = map_ #keep map for motion and measurement updaet
        self._translation_variance = translation_variance #get variance so we can add noise 
        self._rotation_variance = rotation_variance #same but for rotation
        self._measurement_variance = measurement_variance #same but for measurement
        self._n_particles = n_particles #num of particles to initalize
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb #get map bounds
        def valid(x_value: float, y_value: float) -> bool: #see if particle is in valid place
            if not (x_minimum <= x_value <= x_maximum) or not (y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles:
                if (obstacle[0] <= x_value <= obstacle[1]) and (obstacle[2] <= y_value <= obstacle[3]):
                    return False
            return True
        log_p = math.log(1.0 / self._n_particles) #init log prob is same due to uniform distribution
        ampts = 0
        max_ampts = self._n_particles *20 #preventing an infinite loop
        while len(self._particles) < self._n_particles: #add particles til content
            x_value = uniform(x_minimum, x_maximum) #sampling
            y_value = uniform(y_minimum, y_maximum)
            theta_value = uniform(-math.pi, math.pi)
            if valid(x_value, y_value): #if valid add particle
                self._particles.append(Particle(x_value, y_value, theta_value, log_p))
            ampts += 1
        while len(self._particles) < self._n_particles: #add more particles without checking
            self._particles.append(Particle(uniform(x_minimum, x_maximum), uniform(y_minimum, y_maximum),
            uniform(-math.pi,math.pi, log_p)))
        ######### Your code ends here #########

    def visualize_particles(self):
        pa = PoseArray()
        pa.header.frame_id = "odom"
        pa.header.stamp = rospy.Time.now()
        for particle in self._particles:
            pose = Pose()
            pose.position = Point(particle.x, particle.y, 0.01)
            q_np = quaternion_from_euler(0, 0, float(particle.theta))
            pose.orientation = Quaternion(*q_np.tolist())
            pa.poses.append(pose)
        self.particles_visualization_pub.publish(pa)

    def visualize_estimate(self):
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = rospy.Time.now()
        x, y, theta = self.get_estimate()
        pose = Pose()
        pose.position = Point(x, y, 0.01)
        q_np = quaternion_from_euler(0, 0, float(theta))
        pose.orientation = Quaternion(*q_np.tolist())
        ps.pose = pose
        self.estimate_visualization_pub.publish(ps)

    def move_by(self, delta_x, delta_y, delta_theta):
        delta_theta = angle_to_neg_pi_to_pi(delta_theta)

        # Propagate motion of each particle
        ######### Your code starts here #########
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb
        def valid(x_value: float, y_value: float) -> bool: #see if particle is in valid spot again
            if not (x_minimum <= x_value <= x_maximum) or not (y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles: #chekc if particle is in obstacle
                if (obstacle[0] <= x_value <= obstacle[1]) and (obstacle[2] <= y_value <= obstacle[3]):
                    return False
            return True

        for p in self._particles: #move each particle by given delta and add noise as well
            next_x = p.x + delta_x + np.random.normal(0, self._translation_variance)
            next_y = p.y + delta_y + np.random.normal(0, self._translation_variance)
            next_theta = angle_to_neg_pi_to_pi(p.theta + delta_theta + np.random.normal(0, self._rotation_variance))
            if valid(next_x, next_y):
                p.x = next_x
                p.y = next_y
            else: # add some noise for the particles
                p.x = p.x + np.random.normal(0, self._translation_variance)
                p.y = p.y + np.random.normal(0, self._translation_variance)
            p.theta = next_theta
        ######### Your code ends here #########

    def measure(self, z: float, scan_angle_in_rad: float):
        """Update the particles based on the measurement `z` at the given `scan_angle_in_rad`.

        Args:
            z: distance to an obstacle
            scan_angle_in_rad: Angle in the robots frame where the scan was taken
        """

        # Calculate posterior probabilities and resample
        ######### Your code starts here #########
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb
        def valid(x_value: float, y_value: float) -> bool:
            if not (x_minimum <= x_value <= x_maximum) or not (y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles: #
                if (obstacle[0] <= x_value <= obstacle[1]) and (obstacle[2] <= y_value <= obstacle[3]):
                    return False
            return True
        
        log_w_temp = [] #keep log weights of each particle
        for p in self._particles:#for each particles get pre dist and calc likelihood then add likelihood to log prob and add to log w
            if not valid(p.x, p.y):
                log_w_temp.append(-inf)
                continue
            angle_pred = angle_to_neg_pi_to_pi(p.theta + scan_angle_in_rad)#get angle of scan in world frame
            dist_pred = self._map.closest_distance((p.x, p.y), angle_pred)#get pred distance to obstacle from part and angle
            if dist_pred is None or np.isinf(z):
                log_w_temp.append(-np.inf)
                continue
            likelihood = scipy.stats.norm(dist_pred, math.sqrt(self._measurement_variance)).logpdf(z)#get liklihood
            p.log_p += likelihood
            log_w_temp.append(likelihood) #addliklihood to list
        maximum_logw = max(log_w_temp)
        if np.isneginf(maximum_logw):#if all wiehgt are neg inf then reset to uniform dist
            unilogp = math.log(1.0 / self._n_particles)
            for p in self._particles:
                p.log_p = unilogp
            return
        w_temp = [math.exp(log_w - maximum_logw) for log_w in log_w_temp]
        w_sum = sum(w_temp)
        if w_sum == 0 or np.isnan(w_sum): #if all wieghts are zero then reset to uniform dist
            unilogp = math.log(1.0 / self._n_particles)
            for p in self._particles:
                p.log_p = unilogp
            return
        w_temp = [w / w_sum for w in w_temp] #normalization
        resampled_particles = choice(self._particles, size=self._n_particles, replace=True, p=w_temp)
        newp = []
        unilogp = math.log(1.0 / self._n_particles)
        for r in resampled_particles: #get the new particles and reset the log probs to a uniform dist
            #p = r
            newp.append(Particle(r.x, r.y, r.theta, unilogp))
        self._particles = newp
        ######### Your code ends here #########

    def get_estimate(self) -> Tuple[float, float, float]:
        # Estimate robot's location using particle weights
        ######### Your code starts here #########
        particle_logs = np.array([p.log_p for p in self._particles], dtype=np.float64) #get log probs of each particle
        maximum_logp = np.max(particle_logs)
        if np.isneginf(maximum_logp): #if all are neg inf reset to uniform dist and return avg of particles
            weights = np.ones(len(self._particles), dtype=np.float64) / len(self._particles)
        else:
            weights = np.exp(particle_logs - maximum_logp) #subtract max logp for numerical stability then get wieghts by exponentiating
            weights_sum = np.sum(weights)
            if weights_sum == 0 or np.isnan(weights_sum):#if all weights are zero reset to unifrom dist
                weights = np.ones(len(self._particles), dtype=np.float64) / len(self._particles)
            else:
                weights /= weights_sum #normalize
        
        all_x = np.array([p.x for p in self._particles], dtype=np.float64)#get x,y,theta of each particle
        all_y = np.array([p.y for p in self._particles], dtype=np.float64)
        all_theta = np.array([p.theta for p in self._particles], dtype=np.float64)
        x_estimate = np.sum(all_x * weights)#get estimates by weighted avg of all particlaes
        y_estimate = np.sum(all_y * weights)
        theta_estimate = atan2(np.sum(np.sin(all_theta) * weights), np.sum(np.cos(all_theta) * weights))
        return x_estimate, y_estimate, theta_estimate
        ######### Your code ends here #########


class Controller:
    def __init__(self, particle_filter: ParticleFilter):
        rospy.init_node("particle_filter_controller", anonymous=True)
        self._particle_filter = particle_filter
        self._particle_filter.visualize_particles()

        #
        self.current_position = None
        self.laserscan = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)
        self.target_position_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)

        while ((self.current_position is None) or (self.laserscan is None)) and (not rospy.is_shutdown()):
            rospy.loginfo("waiting for odom and laserscan")
            rospy.sleep(0.1)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def visualize_laserscan_ranges(self, idx_groups: List[Tuple[int, int]]):
        """Helper function to visualize ranges of sensor readings from the laserscan lidar.

        Example usage for visualizing the first 10 and last 10 degrees of the laserscan:
            `self.visualize_laserscan_ranges([(0, 10), (350, 360)])`
        """
        pcd = PointCloud()
        pcd.header.frame_id = "odom"
        pcd.header.stamp = rospy.Time.now()
        for idx_low, idx_high in idx_groups:
            for idx, d in enumerate(self.laserscan.ranges[idx_low:idx_high]):
                if d == inf:
                    continue
                angle = math.radians(idx) + self.current_position["theta"]
                x = d * math.cos(angle) + self.current_position["x"]
                y = d * math.sin(angle) + self.current_position["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
        self.pointcloud_pub.publish(pcd)

    def visualize_position(self, x: float, y: float):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(x, y, 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.075, 0.075, 0.1)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
        marker_array.markers.append(marker)
        self.target_position_pub.publish(marker_array)

    def take_measurements(self):
        # Take measurement using LIDAR
        ######### Your code starts here #########
        # NOTE: with more than 2 angles the particle filter will converge too quickly, so with high likelihood the
        # correct neighborhood won't be found.
        angle_range = [0, pi /2] #use front and left of robot to get convergence
        for angle in angle_range: #get dist measurement and update particle filter for each angle
            index = int(round((angle - self.laserscan.angle_min) / self.laserscan.angle_increment))
            index = max(0, min(index, len(self.laserscan.ranges) - 1))
            distance = self.laserscan.ranges[index]
            if np.isinf(distance) or np.isnan(distance):
                continue
            self._particle_filter.measure(distance, angle)
        self._particle_filter.visualize_particles() #methods to deal with particles
        self._particle_filter.visualize_estimate()
        ######### Your code ends here #########

    def autonomous_exploration(self):
        """Randomly explore the environment here, while making sure to call `take_measurements()` and
        `_particle_filter.move_by()`. The particle filter should converge on the robots position eventually.

        Note that the following visualizations functions are available:
            visualize_position(...)
            visualize_laserscan_ranges(...)
        """
        # Robot autonomously explores environment while it localizes itself
        ######### Your code starts here #########
        stable =0
        step= 0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown(): #exploreuntil the convergence is met
            frontindex = int(round((0 - self.laserscan.angle_min) / self.laserscan.angle_increment))
            frontindex = max(0,min(frontindex, len(self.laserscan.ranges)-1))
            frontdistance = self.laserscan.ranges[frontindex]
            if np.isinf(frontdistance) or np.isnan(frontdistance):
                frontdistance = 10.0
            if frontdistance < 0.5:# if obs is close turn
                turn_dir = 1 if step % 2 == 0 else -1
                goal_theta = angle_to_neg_pi_to_pi(self.current_position["theta"] + turn_dir * math.pi / 2)
                self.rotate_action(goal_theta)
            else:
                self.forward_action(0.25)
            self.take_measurements()
            step += 1
            estimate_x, estimate_y, estimate_theta = self._particle_filter.get_estimate()#get estimates
            all_x = np.array([p.x for p in self._particle_filter._particles], dtype=np.float64)
            all_y = np.array([p.y for p in self._particle_filter._particles], dtype=np.float64)
            all_theta = np.array([p.theta for p in self._particle_filter._particles], dtype=np.float64)
            dispersement = np.sqrt(np.var(all_x) + np.var(all_y))#varience of particles in x and y direction
            heading =np.sqrt(np.mean(np.sin(all_theta))**2 + np.mean(np.cos(all_theta))**2)#concet of parts in same direction
            xydistance = np.sqrt((all_x - estimate_x)**2 + (all_y - estimate_y)**2) #dists of particles to estimate
            cluster= np.mean(xydistance < 0.16) #perecntage of particles close to estimate
            print("auto step is taken")
            if dispersement < 0.05 and heading > 0.5 and cluster > 0.9:
                stable += 1
            else:
                stable = 0
            
            if step>=7 or stable >=5:
                print("auto exploration is done")
                break
            rate.sleep()
        ######### Your code ends here #########

    def forward_action(self, distance: float):
        # Robot moves forward by a set amount during manual control
        ######### Your code starts here #########
        begin_x = self.current_position["x"]
        begin_y = self.current_position["y"]
        begin_theta = self.current_position["theta"]

        controller = PIDController(1.45, 0.0, 0.09)
        rate = rospy.Rate(20)
        old_time = time()
        if distance >=0:# 
            direction = 1 #move forward
        else:
            direction = -1 #move backward
        target = abs(distance)#move the same dist
        while not rospy.is_shutdown():
            change_x = self.current_position["x"] - begin_x #get change in x,y,theta
            change_y = self.current_position["y"] - begin_y
            change_theta = angle_to_neg_pi_to_pi(self.current_position["theta"] - begin_theta)
            moved_distance = math.sqrt(change_x**2 + change_y**2)#get the dist moved
            noise = target - moved_distance #
            if noise < 0.009:
                break
            change_time = time() - old_time
            old_time = time()
            control_signal = controller.control(noise, change_time) #add noise to control
            control_signal = max(min(control_signal, 0.16), 0.06)
            control_signal *= direction
            twist = Twist()
            twist.linear.x = control_signal
            twist.angular.z = 0.0
            self.robot_ctrl_pub.publish(twist)
            rate.sleep()
        
        self.robot_ctrl_pub.publish(Twist())
        rospy.sleep(0.1)
        finsihed_x = self.current_position["x"]
        finsihed_y = self.current_position["y"]
        finsihed_theta = self.current_position["theta"]
        self._particle_filter.move_by(finsihed_x - begin_x, finsihed_y - begin_y, 0.0)
        ######### Your code ends here #########

    def rotate_action(self, goal_theta: float):
        # Robot turns by a set amount during manual control
        ######### Your code starts here #########

        begin_theta = self.current_position["theta"]
        controller = PIDController(2.05, 0.0, 0.09)
        rate = rospy.Rate(20)
        old_time = time()
        while not rospy.is_shutdown():# keep truning until 0.025 raidans close enough
            noise = angle_to_neg_pi_to_pi(goal_theta - self.current_position["theta"])
            if abs(noise) < 0.025:
                break
            change_time = time() - old_time
            old_time = time()
            control_signal = controller.control(noise, change_time)
            if abs(control_signal) < 0.16:#make sure we deal with any other issuse
                if control_signal >= 0:
                    control_signal = 0.16
                else:
                    control_signal = -0.16
            control_signal = max(min(control_signal, 0.5), -0.5)
            twist = Twist()#same as old labs
            twist.linear.x = 0.0
            twist.angular.z = control_signal
            self.robot_ctrl_pub.publish(twist)
            rate.sleep()
        self.robot_ctrl_pub.publish(Twist())
        rospy.sleep(0.1)
        finsihed_theta = self.current_position["theta"]
        changed_theta = angle_to_neg_pi_to_pi(finsihed_theta - begin_theta)
        self._particle_filter.move_by(0.0, 0.0, changed_theta)
        ######### Your code ends here ######### 


""" Example usage

rosrun development lab8_9.py --map_filepath src/csci455l/scripts/lab8_9_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]

    map_ = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.1
    rotation_variance = 0.05
    measurement_variance = 0.1
    particle_filter = ParticleFilter(map_, num_particles, translation_variance, rotation_variance, measurement_variance)
    controller = Controller(particle_filter)

    try:
        # Manual control
        goal_theta = 0
        controller.take_measurements()
        while not rospy.is_shutdown():
            print("\nEnter 'a', 'w', 's', 'd' to move the robot:")
            uinput = input("")
            if uinput == "w": # forward
                ######### Your code starts here #########
                controller.forward_action(0.28)
                ######### Your code ends here #########
            elif uinput == "a": # left
                ######### Your code starts here #########
                goal_theta = angle_to_neg_pi_to_pi(controller.current_position["theta"] + math.pi / 2)# add 90 degs to current theta
                controller.rotate_action(goal_theta)
                ######### Your code ends here #########
            elif uinput == "d": #right
                ######### Your code starts here #########
                goal_theta = angle_to_neg_pi_to_pi(controller.current_position["theta"] - math.pi / 2)# subtract 90 degs
                controller.rotate_action(goal_theta)
                ######### Your code ends here #########
            elif uinput == "s": # backwards
                ######### Your code starts here #########
                controller.forward_action(-0.28)
                ######### Your code ends here #########
            elif uinput == "auto":
                print("Switching to auto explore")
                break
            else:
                print("Invalid input")
            ######### Your code starts here #########
            controller.take_measurements()
            controller._particle_filter.visualize_estimate()
            controller._particle_filter.visualize_particles()
            ######### Your code ends here #########

        # Autonomous exploration
        ######### Your code starts here #########
        controller.autonomous_exploration()
        ######### Your code ends here #########

    except rospy.ROSInterruptException:
        print("Shutting down...")
