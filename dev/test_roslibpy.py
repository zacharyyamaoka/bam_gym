#!/usr/bin/env python3

# Launch on Server with:
# ros2 launch rosbridge_server rosbridge_websocket_launch.xml
import roslibpy

host_ip = 'localhost'
port = 9090
client = roslibpy.Ros(host=host_ip, port=port)

print(f'Trying to connect to ROS bridge at {host_ip}:{port}...')
client.run(timeout=3)

print('Is ROS connected?', client.is_connected)
client.terminate()