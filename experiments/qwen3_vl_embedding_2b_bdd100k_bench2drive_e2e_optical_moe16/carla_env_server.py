"""Python 3.8 CARLA route environment served to the Python 3.11 learner.

This module intentionally avoids importing torch, transformers, or project
Settings.  It is launched from the dedicated RFL environment that owns the
CARLA 0.9.15 client extension.
"""

from __future__ import annotations

import argparse
import math
import queue
import random
import traceback
from multiprocessing.connection import Listener

import numpy as np


def _carla_imports():
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    return carla, GlobalRoutePlanner


class CarlaRouteEnvironment:
    def __init__(self, arguments):
        carla, planner_class = _carla_imports()
        self.carla = carla
        self.client = carla.Client(arguments.carla_host, arguments.carla_port)
        self.client.set_timeout(arguments.timeout)
        self.world = self.client.get_world()
        current_map = self.world.get_map().name.rsplit("/", 1)[-1]
        if current_map != arguments.map:
            self.world = self.client.load_world(arguments.map)
        self.original_settings = self.world.get_settings()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = True
        world_settings.fixed_delta_seconds = arguments.fixed_delta
        self.world.apply_settings(world_settings)
        self.map = self.world.get_map()
        self.planner = planner_class(self.map, arguments.route_resolution)
        self.image_size = arguments.image_size
        self.max_episode_steps = arguments.max_episode_steps
        self.target_speed_mps = arguments.target_speed_mps
        self.route_lookahead = arguments.route_lookahead
        self.camera_queue = queue.Queue(maxsize=4)
        self.actors = []
        self.vehicle = None
        self.route = []
        self.route_distances = []
        self.route_index = 0
        self.steps = 0
        self.collision = False
        self.lane_invasion = False
        self.rng = random.Random(42)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(int(seed))
        self._destroy_actors()
        self._drain_camera()
        self.collision = False
        self.lane_invasion = False
        self.steps = 0
        blueprint = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("CARLA map has fewer than two vehicle spawn points")
        for _attempt in range(100):
            start, destination = self.rng.sample(spawn_points, 2)
            route = self.planner.trace_route(start.location, destination.location)
            if len(route) < 80:
                continue
            vehicle = self.world.try_spawn_actor(blueprint, start)
            if vehicle is None:
                continue
            self.vehicle = vehicle
            self.actors.append(vehicle)
            self.route = route
            self._build_route_distances()
            break
        else:
            raise RuntimeError("Could not spawn a vehicle on a sufficiently long route")
        self._attach_sensors()
        for _ in range(4):
            self.world.tick()
        observation = self._observation()
        return observation, {"route_points": len(self.route)}

    def step(self, action):
        steer, throttle, brake = [float(value) for value in action]
        control = self.carla.VehicleControl(
            steer=max(-1.0, min(1.0, steer)),
            throttle=max(0.0, min(1.0, throttle)),
            brake=max(0.0, min(1.0, brake)),
        )
        previous_distance = self.route_distances[self.route_index]
        self.vehicle.apply_control(control)
        self.world.tick()
        self.steps += 1
        self._update_route_index()
        progress = self.route_distances[self.route_index] - previous_distance
        observation = self._observation()
        speed = observation["speed"]
        waypoint = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=False,
            lane_type=self.carla.LaneType.Driving,
        )
        offroad = waypoint is None
        lane_offset = self._lane_offset(waypoint) if waypoint is not None else 10.0
        red = self.vehicle.get_traffic_light_state() == self.carla.TrafficLightState.Red
        red_violation = bool(red and speed > 1.0 and throttle > 0.1)
        route_done = self.route_index >= len(self.route) - 3
        terminated = bool(self.collision or route_done)
        truncated = bool(self.steps >= self.max_episode_steps)
        info = {
            "route_progress": float(progress),
            "speed": float(speed),
            "target_speed": float(self.target_speed_mps),
            "lane_offset": float(lane_offset),
            "collision": bool(self.collision),
            "offroad": bool(offroad),
            "red_light": red_violation,
            "lane_invasion": bool(self.lane_invasion),
            "route_index": int(self.route_index),
            "route_points": len(self.route),
            "route_completed": route_done,
        }
        self.lane_invasion = False
        return observation, float(progress), terminated, truncated, info

    def close(self):
        self._destroy_actors()
        if self.world is not None:
            self.world.apply_settings(self.original_settings)

    def _attach_sensors(self):
        library = self.world.get_blueprint_library()
        camera_bp = library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size))
        camera_bp.set_attribute("image_size_y", str(self.image_size))
        camera_bp.set_attribute("fov", "100")
        camera = self.world.spawn_actor(
            camera_bp,
            self.carla.Transform(
                self.carla.Location(x=1.5, z=2.4),
                self.carla.Rotation(pitch=-8.0),
            ),
            attach_to=self.vehicle,
        )
        camera.listen(self._camera_callback)
        self.actors.append(camera)
        collision = self.world.spawn_actor(
            library.find("sensor.other.collision"),
            self.carla.Transform(),
            attach_to=self.vehicle,
        )
        collision.listen(lambda _event: setattr(self, "collision", True))
        self.actors.append(collision)
        invasion = self.world.spawn_actor(
            library.find("sensor.other.lane_invasion"),
            self.carla.Transform(),
            attach_to=self.vehicle,
        )
        invasion.listen(lambda _event: setattr(self, "lane_invasion", True))
        self.actors.append(invasion)

    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4
        )
        rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
        try:
            self.camera_queue.put_nowait(rgb)
        except queue.Full:
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                pass
            self.camera_queue.put_nowait(rgb)

    def _observation(self):
        try:
            image = self.camera_queue.get(timeout=10.0)
        except queue.Empty:
            raise RuntimeError("Timed out waiting for synchronous front RGB camera")
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        target_index = min(len(self.route) - 1, self.route_index + self.route_lookahead)
        target_location = self.route[target_index][0].transform.location
        local_target = self._world_to_ego(target_location)
        command = self._command_index(self.route[target_index][1])
        return {
            "rgb_front": image,
            "speed": float(speed),
            "command": command,
            "target_point": np.asarray(local_target, dtype=np.float32),
        }

    def _update_route_index(self):
        location = self.vehicle.get_location()
        start = max(0, self.route_index - 3)
        end = min(len(self.route), self.route_index + 30)
        nearest = min(
            range(start, end),
            key=lambda index: self.route[index][0].transform.location.distance(location),
        )
        self.route_index = max(self.route_index, nearest)

    def _build_route_distances(self):
        self.route_distances = [0.0]
        for index in range(1, len(self.route)):
            previous = self.route[index - 1][0].transform.location
            current = self.route[index][0].transform.location
            self.route_distances.append(
                self.route_distances[-1] + previous.distance(current)
            )
        self.route_index = 0

    def _world_to_ego(self, location):
        transform = self.vehicle.get_transform()
        delta_x = location.x - transform.location.x
        delta_y = location.y - transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        return (
            math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
            -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
        )

    def _lane_offset(self, waypoint):
        transform = waypoint.transform
        location = self.vehicle.get_location()
        delta_x = location.x - transform.location.x
        delta_y = location.y - transform.location.y
        right = transform.get_right_vector()
        return delta_x * right.x + delta_y * right.y

    @staticmethod
    def _command_index(option):
        value = int(getattr(option, "value", option))
        return max(0, min(5, value - 1))

    def _drain_camera(self):
        while True:
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                return

    def _destroy_actors(self):
        for actor in reversed(self.actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        self.actors = []
        self.vehicle = None


def serve(arguments):
    environment = CarlaRouteEnvironment(arguments)
    listener = Listener(
        (arguments.bridge_host, arguments.bridge_port),
        authkey=arguments.authkey.encode("utf-8"),
    )
    print(
        "CARLA bridge listening on {}:{} -> CARLA {}:{} map={}".format(
            arguments.bridge_host,
            arguments.bridge_port,
            arguments.carla_host,
            arguments.carla_port,
            arguments.map,
        ),
        flush=True,
    )
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    operation = request.get("op")
                    if operation == "hello":
                        result = {"protocol_version": 1, "environment": "carla_route"}
                    elif operation == "reset":
                        observation, info = environment.reset(request.get("seed"))
                        result = {"observation": observation, "info": info}
                    elif operation == "step":
                        observation, reward, terminated, truncated, info = environment.step(
                            request["action"]
                        )
                        result = {
                            "observation": observation,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        }
                    elif operation == "close":
                        connection.send({"ok": True, "result": {}})
                        break
                    else:
                        raise ValueError("Unsupported bridge operation {!r}".format(operation))
                    connection.send({"ok": True, "result": result})
            except EOFError:
                pass
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
            finally:
                connection.close()
    finally:
        environment.close()
        listener.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=24515)
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=24615)
    parser.add_argument("--authkey", default="bench2drive-local")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--target-speed-mps", type=float, default=8.0)
    parser.add_argument("--route-resolution", type=float, default=2.0)
    parser.add_argument("--route-lookahead", type=int, default=5)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
