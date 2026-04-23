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
    # ----------------------------------------------------------------------
    # Real-robot safe laser helpers
    # ----------------------------------------------------------------------
    def _valid_range(self, r):
        """Real TurtleBot3 LDS returns 0.0 or NaN for bad beams; filter those."""
        if r is None:
            return False
        if np.isinf(r) or np.isnan(r):
            return False
        if r < 0.12:  # LDS-01 physical minimum is ~12 cm
            return False
        return True
    
    def _front_min_range(self, half_window_deg=25):
        """Return the minimum VALID range in a ±half_window_deg cone ahead of the
        robot. Auto-detects whether the scan is sim-style (angle_min ≈ -π, front at
        middle of array) or real-style (angle_min ≈ 0, front at index 0)."""
        if self.laserscan is None:
            return float("inf")
        ranges = self.laserscan.ranges
        n = len(ranges)
        if n == 0:
            return float("inf")
    
        if abs(self.laserscan.angle_min) < 0.1:
            # Real robot: index 0 is front, wrap around both sides
            beams_per_deg = n / 360.0
            k = int(round(half_window_deg * beams_per_deg))
            idxs = list(range(0, k + 1)) + list(range(n - k, n))
        else:
            # Sim: front is at the middle of the ranges array
            mid = n // 2
            beams_per_rad = 1.0 / self.laserscan.angle_increment
            k = int(round(math.radians(half_window_deg) * beams_per_rad))
            idxs = list(range(max(0, mid - k), min(n, mid + k + 1)))
    
        valid = [ranges[i] for i in idxs if self._valid_range(ranges[i])]
        return min(valid) if valid else float("inf")
    
    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (overrides Lab 8/9 exploration)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Exploration tuned for the real robot:
          - Index-based front detection that works for sim AND real robot.
          - Filter 0.0 / NaN artifacts from real LDS.
          - No 'back up' fallback — the robot only turns or moves forward,
            never reverses into something it can't see.
        Continues to call take_measurements() so the particle filter keeps
        converging during exploration.
        """
        ######### Your code starts here #########
        rate = rospy.Rate(2.0)
        rotation_streak = 0
        min_steps_before_convergence = 12
    
        for step in range(max_steps):
            if rospy.is_shutdown():
                break
    
            front_dist = self._front_min_range(half_window_deg=25)
            rospy.loginfo(f"[localize {step}] front_dist={front_dist:.2f}")
    
            # Escape if we've been rotating too long
            if rotation_streak > 5:
                rospy.loginfo("Stuck rotating; forcing small forward move.")
                self.move_forward(0.10)
                rotation_streak = 0
    
            elif front_dist < 0.35:
                # Obstacle ahead — turn, don't back up
                self.rotate_in_place(uniform(math.pi / 4, math.pi / 2))
                rotation_streak += 1
    
            else:
                # Clear — go forward
                self.move_forward(0.18)
                rotation_streak = 0
    
            # Update particle filter with the current scan
            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()
    
            # Only check convergence after some real motion
            if step >= min_steps_before_convergence:
                particles = np.array([[p.x, p.y] for p in self._pf._particles])
                x_est, y_est, _ = self._pf.get_estimate()
                spread = float(np.std(
                    np.linalg.norm(particles - np.array([x_est, y_est]), axis=1)
                ))
                rospy.loginfo(f"[localize {step}] spread={spread:.3f}")
                if spread < 0.15:
                    rospy.loginfo("Particle filter converged.")
                    break
    
            rate.sleep()
        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        # get robot's est pos from pf
        x_est, y_est, theta_est = self._pf.get_estimate()
        
        # rospy.loginfo(f"PF estimate for RRT start: ({x_est:.3f}, {y_est:.3f}, {theta_est:.3f})") # DEBUG
        
        start_position = {"x": x_est, "y": y_est, "theta": theta_est}
        
        # run RRT from curr location to goal
        plan, graph = self._planner.generate_plan(start_position, self.goal_position)
        
        # draw it in RViz
        self._planner.visualize_plan(plan)
        self._planner.visualize_graph(graph)

        # error prevention
        if len(plan) == 0:
            # rospy.logwarn("RRT failed to find a path. Retrying.") # DEBUG
            x_est, y_est, theta_est = self._pf.get_estimate()
            start_position = {"x": x_est, "y": y_est, "theta": theta_est}
            plan, graph = self._planner.generate_plan(start_position, self.goal_position)
            self._planner.visualize_plan(plan)
            self._planner.visualize_graph(graph)

        # more error prevention - will this mess up?
        if len(plan) == 0:
            rospy.logerr("RRT could not find a path after retry. Follow phase will be skipped.")
        else:
            rospy.loginfo(f"RRT found a plan with {len(plan)} waypoints.")
        
        # store the plan for the next phase
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
            if self.current_position is None:
                rate.sleep()
                continue
    
            # all waypoints hit —> stop the robot
            if self.current_wp_idx >= len(self.plan):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.cmd_pub.publish(ctrl_msg)
                rospy.loginfo("Goal reached! Robot stopped.")
                break
    
            # get curr waypoint target
            goal = self.plan[self.current_wp_idx]
    
            # get distance & angle error to curr waypoint
            dx = goal["x"] - self.current_position["x"]
            dy = goal["y"] - self.current_position["y"]
            distance_error = sqrt(dx**2 + dy**2)
            target_theta = atan2(dy, dx)
            angle_error = target_theta - self.current_position["theta"]
            angle_error = atan2(math.sin(angle_error), math.cos(angle_error)) # normalize

            # PID control signals
            t = rospy.get_time()
            linear_vel = self.linear_pid.control(distance_error, t)
            angular_vel = self.angular_pid.control(angle_error, t)
    
            # if big error, stop moving and rotate first
            if abs(angle_error) > 0.5:
                linear_vel = 0.0
    
            ctrl_msg.linear.x = linear_vel
            ctrl_msg.angular.z = angular_vel
            self.cmd_pub.publish(ctrl_msg)
    
            # go to next waypoint once robot is close enough
            if distance_error < GOAL_THRESHOLD:
                rospy.loginfo(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.plan)}: "
                              f"({goal['x']:.2f}, {goal['y']:.2f})")
                self.current_wp_idx += 1
    
            # keep pf updated while moving
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
