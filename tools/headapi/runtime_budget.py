"""Standard-library host RAM guard; no model SDK imported by API process."""
import ctypes
from pathlib import Path
import subprocess
import sys,time


def available_ram():
    if sys.platform=='win32':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
        status=Status();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise OSError('Cannot read host RAM')
        return int(status.available)
    if sys.platform.startswith('linux'):
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return None


def run_guarded(args,*,cwd,env,stdout,timeout,minimum,abort):
    free=available_ram()
    if free is not None and free<minimum:raise MemoryError('INSUFFICIENT_HOST_MEMORY_BEFORE_STAGE')
    child=subprocess.Popen(args,cwd=cwd,env=env,stdout=stdout,stderr=subprocess.STDOUT)
    started=time.monotonic()
    try:
        while child.poll() is None:
            free=available_ram()
            if free is not None and free<abort:raise MemoryError('HOST_MEMORY_BUDGET_EXCEEDED')
            if time.monotonic()-started>timeout:raise subprocess.TimeoutExpired(args,timeout)
            time.sleep(.5)
        return subprocess.CompletedProcess(args,child.returncode)
    finally:
        if child.poll() is None:
            if sys.platform=='win32':
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            else:child.terminate()
            child.wait()
