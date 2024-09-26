import datetime
import os
import time
import subprocess
import importlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


FINISHED_RUNS_PATH = "./runs"

log_file_path = "./runs/runner_log.txt"
log_file = open(log_file_path, "w")


def log_and_save(message):
    log_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
    # log_file.write(message + "\n")
    log_file.flush()  # Ensure it's written to the file immediately


def trynmove(file):
    filename = file.split("/")[-1]
    filename_without_py = filename[:-3]
    try:
        os.rename(file, "./runs/" + filename_without_py + "/" + filename)
    except:
        os.rename(file, file[:-3] + "_rename_error.txt")


def run_sub_for_all_python_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                try:
                    print("importing")
                    importlib.import_module("RUNNER_FOLDER." + file[:-3], package=None)
                    log_and_save(f"Executed and deleted: {file}")
                    trynmove("./RUNNER_FOLDER/" + file)
                except Exception as e:
                    os.rename(file, file[:-3] + "_error.txt")
                    log_and_save(f"Error executing {file}: {e}")


class ScriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        if event.src_path.endswith(".py"):
            try:
                file = event.src_path
                filename = file.split("/")[-1]
                filename_without_py = filename[:-3]
                print("importing based on observer")
                importlib.import_module(
                    "RUNNER_FOLDER." + filename_without_py, package=None
                )
                log_and_save(f"Executed and deleted: {event.src_path}")
                trynmove(str(event.src_path))
            except Exception as e:
                os.rename(event.src_path, str(event.src_path)[:-3] + "_error.txt")
                log_and_save(f"Error executing {event.src_path}: {e}")


if __name__ == "__main__":
    path_to_watch = "./RUNNER_FOLDER"
    run_sub_for_all_python_files_in_directory(path_to_watch)

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
