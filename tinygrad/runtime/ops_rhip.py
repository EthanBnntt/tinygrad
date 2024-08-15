import ctypes, os
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.compiler_hip import compile_hip

rhip = ctypes.CDLL(os.getenv("RHIP_PATH", "")+"/target/release/libremu.dylib")
class AMDCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

class RHIPProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
  def __call__(self, *args, global_size, local_size, vals=(), wait=False):
    args = (*args, *vals)
    rhip.run_asm(self.lib, len(self.lib), *global_size, *local_size, (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

os.environ["OSX"] = "1"
class RHIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(device, MallocAllocator, HIPRenderer(), AMDCompiler("gfx1100"), RHIPProgram)
