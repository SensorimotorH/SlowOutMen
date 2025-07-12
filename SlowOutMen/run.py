from SlowOutMen.TaskUnderstanding.task_parser import TaskParser
from SlowOutMen.Communication.ros2_mmk2_api import MMK2_Receiver
import rclpy
import subprocess

if __name__ == "__main__":
    
    rclpy.init()
    receiver = MMK2_Receiver()
    task_parser = TaskParser()
    instruction = receiver.get_taskinfo()
    task_key = task_parser.parse_task(instruction)
    
    if task_key["round"] == 1:
        try:
            print("round 1")
            command = f"cd /workspace/SlowOutMen && python3 round_1.py"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"Error executing command: {e}")
            pass
        game_info = receiver.get_gameinfo()
        print(game_info)
    elif task_key["round"] == 2:
        try:
            print("round 2")
            command = f"cd /workspace/SlowOutMen && python3 round_2.py"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            pass
    elif task_key["round"] == 3:
        try:
            print("round 3")
            command = f"cd /workspace/SlowOutMen && python3 round_3.py"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:  
            pass