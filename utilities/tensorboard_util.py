import os
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def launch_tensorboard(logdir="logs/pretrain/", port=6007):
    try:
        # Kill any process using the specified port (for clean restart)
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Start TensorBoard
        subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", str(port)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print(f"TensorBoard started at http://localhost:{port}/")
    
    except Exception as e:
        print(f"Error starting TensorBoard: {e}")