import os.path
import time
import threading
from datetime import datetime
SLEEP = 1

KEEP_GOING = threading.Event()


def grep_for_nvidia_errors(file, offset):
    ret = list()
    with open(file, 'r') as fp:
        # It will start with zero on the first time
        fp.seek(offset)
        for line in fp:
            line_lower = line.lower()
            if "nvidia" in line_lower or "nvrm" in line_lower:
                ret.append(line)
            # Increase to know where to jump after
            offset += len(line)
    return ret, offset


def read_system_logs():
    global KEEP_GOING, SLEEP
    var_dir_logs = "/var/log"
    # The file and its offset
    system_log_files = {
        "syslog": 0, "kern.log": 0
    }
    out_file = "/var/radiation-benchmarks/log/sys_logs.log"

    # The thread will work until ctrl-c is pressed
    while KEEP_GOING.is_set() is not True:
        for log_file in system_log_files:
            sys_log_file = f"{var_dir_logs}/{log_file}"
            searched_list, system_log_files[log_file] = grep_for_nvidia_errors(
                sys_log_file, system_log_files[log_file])
            with open(out_file, "a") as out_fp:
                out_fp.writelines(searched_list)
        file_size = os.path.getsize(out_file) / 1024 ** 2
        print(f"Updating the system log size {file_size} at timestamp {datetime.fromtimestamp(time.time())}")
        if (file_size / 1024.0) > 2:
            break
        KEEP_GOING.wait(timeout=SLEEP)


log_thread = threading.Thread(target=read_system_logs)
try:
    print("Starting thread")
    log_thread.start()
except KeyboardInterrupt:
    KEEP_GOING.set()
    log_thread.join()
