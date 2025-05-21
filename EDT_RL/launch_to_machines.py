import time

import paramiko
import os
import threading
import json
import io
from tqdm import tqdm
from scp import SCPClient
import concurrent.futures

PROJECT_NAME = "EDT_RL"
BASE_DIR = f"{PROJECT_NAME}"
LOCAL_PACKAGE = "./train"
MACHINES_FILE = "machines.json"


def upload_directory(client, local_dir, remote_dir):
    sftp = client.open_sftp()
    try:
        # Ensure remote_dir exists
        sftp.mkdir(remote_dir)
    except IOError:
        pass  # Directory may already exist
    sftp.close()

    scp = SCPClient(client.get_transport())
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        if os.path.isdir(local_path):
            scp.put(local_path, recursive=True, remote_path=remote_dir)
        else:
            scp.put(local_path, remote_path=remote_dir)
    scp.close()


def stream_output(stdout, stderr):
    def stream_pipe(pipe):
        for line in iter(pipe.readline, ""):
            print(line, end="")

    stdout_thread = threading.Thread(target=stream_pipe, args=(stdout,))
    stderr_thread = threading.Thread(target=stream_pipe, args=(stderr,))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()


def run(client, pc_id):
    command_to_run = "python3.11 runner.py"
    tmux_session_name = "rl_session"

    # clear tmux session
    stdin, stdout, stderr = client.exec_command(f'tmux ls')
    print("tmux ls:", stdout.read().decode())
    stdin, stdout, stderr = client.exec_command(f'tmux kill-session -t {tmux_session_name}')
    time.sleep(1)
    stdin, stdout, stderr = client.exec_command(f'tmux ls')

    print("tmux ls:", stdout.read().decode())

    stdin, stdout, stderr = client.exec_command(
        f"cd {BASE_DIR}/{pc_id} &&  tmux new-session -d -s {tmux_session_name} \"{command_to_run}\""
    )
    stream_output(stdout, stderr)


def launch(username: str, password: str, pc_id: str):
    hostname = f"{pc_id}.utm.utoronto.ca"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    stdin, stdout, stderr = client.exec_command("hostname -I")
    ip_output = stdout.read().decode().strip()
    ip_output = ip_output.split()[0]
    print(f"{hostname}: {ip_output}")

    # stdin, stdout, stderr = client.exec_command("nvidia-smi")
    # print(stdout.read().decode().strip())

    # client.exec_command(f"rm -rf {BASE_DIR}")
    client.exec_command(f"mkdir -p {BASE_DIR}")
    upload_directory(client, LOCAL_PACKAGE, f"{BASE_DIR}/{pc_id}")

    server_thread = threading.Thread(target=run, args=(client, pc_id))
    server_thread.start()
    return client, server_thread, ip_output


if __name__ == "__main__":
    password = os.getenv("password")
    with open(MACHINES_FILE, "r") as f:
        MACHINES = json.load(f)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(launch, "your_user_name", password, f"{machine}") for machine in MACHINES]
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Launching..."):
            pass
