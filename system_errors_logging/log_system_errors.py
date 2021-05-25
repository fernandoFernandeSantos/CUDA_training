#!/usr/bin/python3
import configparser
import os
import time
from datetime import datetime

CONF_FILE = '/etc/radiation-benchmarks.conf'
# Sleep on event
SLEEP = 1
# Max GB allowed
MAX_MB_ALLOWED = 2048
SEARCH_NVIDIA_STRINGS = {"nvidia", "nvrm"}
# In seconds
PRINTING_SECONDS_INTERVAL = 120


def grep_for_nvidia_errors(file, offset):
    ret = list()
    with open(file, 'r') as fp:
        # It will start with zero on the first time
        fp.seek(offset)
        for line in fp:
            line_lower = line.lower()
            if any([i for i in SEARCH_NVIDIA_STRINGS if i in line_lower]):
                ret.append(line)
            # Increase to know where to jump after
            offset += len(line)
    return ret, offset


def get_radiation_benchmarks_log_dir():
    try:
        config = configparser.RawConfigParser()
        config.read(CONF_FILE)

        var_dir = config.get('DEFAULT', 'vardir') + "/"
        return var_dir
    except IOError as e:
        raise IOError("System configuration setup error: " + str(e))


def main():
    global SLEEP
    var_dir_logs = "/var/log"
    rad_var_dir = get_radiation_benchmarks_log_dir()
    # The file and its offset
    system_log_files = {
        "syslog": 0, "kern.log": 0
    }

    sys_log_output_file_name = datetime.fromtimestamp(int(time.time())).strftime("%Y_%m_%d_%H_%M_%S")
    sys_log_output_file_name = f"{rad_var_dir}/log/{sys_log_output_file_name}_sys_logs.log"

    # The thread will work until ctrl-c is pressed
    last_print = 0
    while True:
        for log_file in system_log_files:
            sys_log_file = f"{var_dir_logs}/{log_file}"
            searched_list, system_log_files[log_file] = grep_for_nvidia_errors(
                sys_log_file, system_log_files[log_file])
            with open(sys_log_output_file_name, "a") as out_fp:
                out_fp.writelines(searched_list)
        # Convert to MB
        file_size = os.path.getsize(sys_log_output_file_name) / 1024 ** 2
        if file_size > MAX_MB_ALLOWED:
            break

        if last_print == 0:
            print(f"Updating the system log file {sys_log_output_file_name}"
                  f" size {file_size:.3}MB at timestamp {datetime.fromtimestamp(int(time.time()))}")
        last_print = (last_print + 1) % (PRINTING_SECONDS_INTERVAL / SLEEP)
        time.sleep(SLEEP)


if __name__ == '__main__':
    main()
