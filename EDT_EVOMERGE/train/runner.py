from flask import Flask, request
import subprocess, threading, datetime, os
import torch
import gc

app = Flask(__name__)
process = None
current_command = None
current_status = "FAILED"
is_running = False


@app.route("/", methods=["GET"])
def get_subprocess_status():
    return {"command": current_command, "status": current_status}


@app.route("/", methods=["POST"])
def run_command():
    global process, current_command, current_status, is_running
    command = request.json.get("command")
    if not command:
        return {"error": "No command provided"}, 400
    if is_running:
        return {"error": "Process already running"}, 400

    current_command = command
    current_status = "RUNNING"
    is_running = True

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, "log.txt")
    with open(log_file_path, "w"):
        pass  # clear log

    def run_subproc():
        global current_status, is_running
        with open(log_file_path, "a") as log_file:
            proc = subprocess.Popen(command, shell=True, stdout=log_file, stderr=log_file)
            return_code = proc.wait()
            current_status = "SUCCESS" if return_code == 0 else "FAILED"
            is_running = False
            est = datetime.timezone(datetime.timedelta(hours=-5))  # EST is UTC-5
            current_time = datetime.datetime.now(est)
            filename = os.path.join(log_dir, f"log_{current_time.strftime('%Y%m%d_%H_%M_%S')}.txt")
            os.rename(log_file_path, filename)
            open(log_file_path, "w").close()
            torch.cuda.empty_cache()
            gc.collect()

    process = threading.Thread(target=run_subproc)
    process.start()
    return {"message": "Subprocess started", "status": current_status}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7899)
