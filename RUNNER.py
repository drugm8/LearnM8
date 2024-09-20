import datetime
import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


FINISHED_RUNS_PATH = "./runs"

log_file_path = "./runs/runner_log.txt"
log_file = open(log_file_path, "w")

def log_and_save(message):
    log_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
    #log_file.write(message + "\n")
    log_file.flush()  # Ensure it's written to the file immediately


class ScriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        if event.src_path.endswith('.py'):
            try:
                env_name = "threetwelve"  # Replace with your actual environment name
                subprocess.run(["conda", "run", "-n", env_name, "python", event.src_path], check=True)
                log_and_save(f"Executed and deleted: {event.src_path}")
                os.rename(event.src_path, str(event.src_path)[:-3] + "_done.txt")
            except subprocess.CalledProcessError as e:
                os.rename(event.src_path, str(event.src_path)[:-3] + "_error.txt")
                log_and_save(f"Error executing {event.src_path}: {e}")

if __name__ == "__main__":
    path_to_watch = "./RUNNER_FOLDER"
    event_handler = ScriptHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(10) 
  # Adjust sleep interval if needed
    except KeyboardInterrupt:
        observer.stop()
    observer.join()