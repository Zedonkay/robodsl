import ctypes
import json
import os

_lib = None
def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    # Try to find the shared library in the same directory as this file
    libname = "librobodsl_typecheck.so"
    if os.uname().sysname == "Darwin":
        libname = "librobodsl_typecheck.dylib"
    libpath = os.path.join(os.path.dirname(__file__), libname)
    _lib = ctypes.CDLL(libpath)
    _lib.check_ros2_type.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    _lib.check_ros2_type.restype = ctypes.c_bool
    return _lib

def check_ros2_type(type_str, value):
    lib = _load_lib()
    value_json = json.dumps(value)
    error_buf = ctypes.create_string_buffer(256)
    debug_buf = ctypes.create_string_buffer(256)
    ok = lib.check_ros2_type(type_str.encode(), value_json.encode(), error_buf, 256, debug_buf, 256)
    debug = debug_buf.value.decode()
    if debug:
        print(f"[typecheck debug] {debug.strip()}")
    return ok, error_buf.value.decode(), debug 