#!/usr/bin/env python3
"""
============================================================================
 Challenge 5 — Hand Teleop Bridge (runs on REMOTE PC / Docker container)
============================================================================

 Connects to the local_hand_control.py TCP server running on the host
 machine (your laptop) via host.docker.internal, receives velocity
 commands, and publishes Twist messages to /cmd_vel.

 This node does NOT use mediapipe or the camera — all vision processing
 happens on the laptop.

 RUN
 ---
   ros2 run my_cv_package hand_teleop
   (first start local_hand_control.py on your laptop)

   Optional parameters:
     --ros-args -p host:=host.docker.internal -p port:=9090
============================================================================
"""

import json
import socket
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class HandTeleopBridge(Node):

    def __init__(self):
        super().__init__('hand_teleop')

        # host.docker.internal resolves to the Docker host (your laptop)
        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 9090)
        self.declare_parameter('max_linear', 0.18)
        self.declare_parameter('max_angular', 0.8)

        self.host    = self.get_parameter('host').value
        self.port    = self.get_parameter('port').value
        self.max_lin = self.get_parameter('max_linear').value
        self.max_ang = self.get_parameter('max_angular').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._v = 0.0
        self._w = 0.0
        self._lock = threading.Lock()

        # Publish at 20 Hz
        self.create_timer(0.05, self._publish_loop)

        # TCP client in background thread
        self._client_thread = threading.Thread(target=self._tcp_client, daemon=True)
        self._client_thread.start()

        self.get_logger().info(
            f'Hand teleop bridge started. Connecting to {self.host}:{self.port} ...')

    def _publish_loop(self):
        with self._lock:
            v, w = self._v, self._w
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def _tcp_client(self):
        """Connect to the local hand control server and read commands."""
        while rclpy.ok():
            try:
                self.get_logger().info(f'Connecting to {self.host}:{self.port} ...')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)
                sock.connect((self.host, self.port))
                sock.settimeout(2.0)
                self.get_logger().info(f'Connected to {self.host}:{self.port}!')
                self._handle_connection(sock)
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                self.get_logger().warn(
                    f'Cannot connect to {self.host}:{self.port} — {e}. '
                    'Make sure local_hand_control.py is running on your laptop. '
                    'Retrying in 3s...')
            finally:
                # Safety: stop robot on disconnect
                with self._lock:
                    self._v = 0.0
                    self._w = 0.0
                self.get_logger().info('Disconnected — robot stopped. Retrying...')
            time.sleep(3.0)

    def _handle_connection(self, sock):
        """Read newline-delimited JSON messages."""
        buf = ''
        while rclpy.ok():
            try:
                data = sock.recv(4096)
                if not data:
                    break
                buf += data.decode('utf-8', errors='ignore')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        v = max(-self.max_lin, min(self.max_lin, msg.get('v', 0.0)))
                        w = max(-self.max_ang, min(self.max_ang, msg.get('w', 0.0)))
                        with self._lock:
                            self._v = v
                            self._w = w
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError):
                break
        sock.close()

    def destroy_node(self):
        t = Twist()
        self.cmd_pub.publish(t)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandTeleopBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()