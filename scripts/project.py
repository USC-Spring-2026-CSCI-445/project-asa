#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab8_9_starter import Controller # AMISHA ADDED
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # AMISHA ADDED
        self._particle_filter = self._pf            # AMISHA ADDED
        self.forward_action = self.move_forward     # AMISHA ADDED
        self.rotate_action = self.rotate_in_place   # AMISHA ADDED

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        offset = int(15.0 / (angle_increment * 180.0 / math.pi))  # 15 degrees offset

        indices = [max(0, min(num_ranges - 1, mid_idx + i)) for i in (-offset, 0, offset)]
        measurements = []

        for idx in indices:
            z = ranges[idx]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0  # fallback
            angle = angle_min + idx * angle_increment  # angle in robot frame
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        
        ######### Your code starts here #########
        Controller.autonomous_exploration(self)
        
        # particlesLocalized = False
        # turn_direction = 1
        # extra_steps = 0
        # step = 0
    
        # while not rospy.is_shutdown() and step < max_steps:
        #     # --- Check front cone of LiDAR for nearby obstacles ---
        #     cone = 15
        #     front_ranges = list(self.laserscan.ranges[:cone]) + list(self.laserscan.ranges[-cone:])
        #     front_ranges = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]
        #     min_front_dist = min(front_ranges) if front_ranges else float('inf')
    
        #     # --- Motion policy ---
        #     if min_front_dist < 0.5:
        #         # Obstacle close ahead: turn away (alternate direction to avoid getting stuck)
        #         turn_direction *= -1
        #         self.rotate_in_place(turn_direction * pi / 2)
        #     else:
        #         # Path is clear: move forward
        #         self.move_forward(0.3)
    
        #     # --- PF update + visualization ---
        #     self.take_measurements()
        #     self._pf.visualize_particles()
        #     self._pf.visualize_estimate()
    
        #     # --- Convergence check: measure how tight the particle cloud is ---
        #     xs = np.array([p.x for p in self._pf._particles])
        #     ys = np.array([p.y for p in self._pf._particles])
        #     spread = math.sqrt(np.var(xs) + np.var(ys))  # "radius" of cloud in meters
        #     rospy.loginfo(f"[Step {step}] Particle spread: {spread:.3f}")
    
        #     if not particlesLocalized:
        #         # Latch to True once cloud has collapsed below threshold
        #         if spread < 0.25:
        #             particlesLocalized = True
        #             rospy.loginfo("Particles converged. Verifying for 15 more steps...")
        #     else:
        #         # Already converged — run a few more steps to confirm stability, then exit
        #         extra_steps += 1
        #         if extra_steps >= 15:
        #             break
    
        #     step += 1
    
        # x, y, th = self._pf.get_estimate()
        # rospy.loginfo(f"Localized at ({x:.2f}, {y:.2f}, {th:.2f})")

        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        # Step 1: Get the robot's estimated position from the particle filter
        x_est, y_est, theta_est = self._pf.get_estimate()
        
        rospy.loginfo(f"PF estimate for RRT start: ({x_est:.3f}, {y_est:.3f}, {theta_est:.3f})")
        
        start_position = {"x": x_est, "y": y_est, "theta": theta_est}
        
        # Step 2: Run RRT from estimated start to known goal
        plan, graph = self._planner.generate_plan(start_position, self.goal_position)
        
        # Step 3: Visualize the plan and graph in RViz
        self._planner.visualize_plan(plan)
        self._planner.visualize_graph(graph)
        
        if len(plan) == 0:
            rospy.logwarn("RRT failed to find a path! Retrying once...")
            # Retry once — PF estimate may have been noisy
            x_est, y_est, theta_est = self._pf.get_estimate()
            start_position = {"x": x_est, "y": y_est, "theta": theta_est}
            plan, graph = self._planner.generate_plan(start_position, self.goal_position)
            self._planner.visualize_plan(plan)
            self._planner.visualize_graph(graph)
        
        if len(plan) == 0:
            rospy.logerr("RRT could not find a path after retry. Follow phase will be skipped.")
        else:
            rospy.loginfo(f"RRT found a plan with {len(plan)} waypoints.")
        
        # Step 4: Store the plan for follow_plan() to consume
        self.plan = plan
        self.current_wp_idx = 0

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########
        if not self.plan or len(self.plan) == 0:
            rospy.logerr("No plan to follow! Did plan_with_rrt() succeed?")
            return
    
        rospy.loginfo(f"Following plan with {len(self.plan)} waypoints.")
        rate = rospy.Rate(20)  # 20 Hz, matching lab10 controller
        ctrl_msg = Twist()
        self.current_wp_idx = 0
    
        while not rospy.is_shutdown():
    
            # Wait for odom to come in
            if self.current_position is None:
                rate.sleep()
                continue
    
            # All waypoints reached — stop the robot
            if self.current_wp_idx >= len(self.plan):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.cmd_pub.publish(ctrl_msg)
                rospy.loginfo("Goal reached! Robot stopped.")
                break
    
            # Get current waypoint target
            goal = self.plan[self.current_wp_idx]
    
            # Calculate distance and angle error to current waypoint
            dx = goal["x"] - self.current_position["x"]
            dy = goal["y"] - self.current_position["y"]
            distance_error = sqrt(dx**2 + dy**2)
            target_theta = atan2(dy, dx)
            angle_error = target_theta - self.current_position["theta"]
            # Normalize angle error to [-pi, pi]
            angle_error = atan2(math.sin(angle_error), math.cos(angle_error))
    
            # Compute PID control signals
            t = rospy.get_time()
            linear_vel = self.linear_pid.control(distance_error, t)
            angular_vel = self.angular_pid.control(angle_error, t)
    
            # If robot is significantly misaligned, stop moving forward and rotate first
            if abs(angle_error) > 0.5:
                linear_vel = 0.0
    
            ctrl_msg.linear.x = linear_vel
            ctrl_msg.angular.z = angular_vel
            self.cmd_pub.publish(ctrl_msg)
    
            # Advance to next waypoint once close enough
            if distance_error < GOAL_THRESHOLD:
                rospy.loginfo(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.plan)}: "
                              f"({goal['x']:.2f}, {goal['y']:.2f})")
                self.current_wp_idx += 1
    
            # Keep PF updated while moving
            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()
    
            rate.sleep()

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.35

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
