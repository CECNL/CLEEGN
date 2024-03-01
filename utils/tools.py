import traceback
import json
import sys
import os

def CC():
    sys.stderr.write("\x1b[1;31m  ***** hit checkpoint *\n\n\x1b[0m")
    exit(0)

def assert_(boolean_t, assert_msg=""):
    stk_trace = traceback.format_stack()[:-1]
    try:
        assert boolean_t, assert_msg
    except:
        err_trace = traceback.format_exc().strip().split("\n")
        sys.stderr.write("\x1b[0;33m" + err_trace[0] + "\n")
        [sys.stderr.write(_) for _ in stk_trace]
        [sys.stderr.write(_ + "\n") for _ in err_trace[1: -1]]
        sys.stderr.write(f"\x1b[1;31m{err_trace[-1]}\n\x1b[0m")
        exit(1)

def read_json(filepath):
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content
