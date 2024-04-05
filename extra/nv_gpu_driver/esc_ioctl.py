# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/src/common/sdk/nvidia/inc', '-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/src/nvidia/arch/nvalloc/unix/include', '-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/src/common/sdk/nvidia/inc/ctrl']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes

def BIT(x): return 1 << x
NVBIT32 = BIT

class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



NV_ESCAPE_H_INCLUDED = True # macro
NV_ESC_RM_ALLOC_MEMORY = 0x27 # macro
NV_ESC_RM_ALLOC_OBJECT = 0x28 # macro
NV_ESC_RM_FREE = 0x29 # macro
NV_ESC_RM_CONTROL = 0x2A # macro
NV_ESC_RM_ALLOC = 0x2B # macro
NV_ESC_RM_CONFIG_GET = 0x32 # macro
NV_ESC_RM_CONFIG_SET = 0x33 # macro
NV_ESC_RM_DUP_OBJECT = 0x34 # macro
NV_ESC_RM_SHARE = 0x35 # macro
NV_ESC_RM_CONFIG_GET_EX = 0x37 # macro
NV_ESC_RM_CONFIG_SET_EX = 0x38 # macro
NV_ESC_RM_I2C_ACCESS = 0x39 # macro
NV_ESC_RM_IDLE_CHANNELS = 0x41 # macro
NV_ESC_RM_VID_HEAP_CONTROL = 0x4A # macro
NV_ESC_RM_ACCESS_REGISTRY = 0x4D # macro
NV_ESC_RM_MAP_MEMORY = 0x4E # macro
NV_ESC_RM_UNMAP_MEMORY = 0x4F # macro
NV_ESC_RM_GET_EVENT_DATA = 0x52 # macro
NV_ESC_RM_ALLOC_CONTEXT_DMA2 = 0x54 # macro
NV_ESC_RM_ADD_VBLANK_CALLBACK = 0x56 # macro
NV_ESC_RM_MAP_MEMORY_DMA = 0x57 # macro
NV_ESC_RM_UNMAP_MEMORY_DMA = 0x58 # macro
NV_ESC_RM_BIND_CONTEXT_DMA = 0x59 # macro
NV_ESC_RM_EXPORT_OBJECT_TO_FD = 0x5C # macro
NV_ESC_RM_IMPORT_OBJECT_FROM_FD = 0x5D # macro
NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO = 0x5E # macro
NV_ESC_RM_LOCKLESS_DIAGNOSTIC = 0x5F # macro
NV_IOCTL_H = True # macro
NV_IOCTL_NUMBERS_H = True # macro
NV_IOCTL_MAGIC = 'F' # macro
NV_IOCTL_BASE = 200 # macro
NV_ESC_CARD_INFO = (200+0) # macro
NV_ESC_REGISTER_FD = (200+1) # macro
NV_ESC_ALLOC_OS_EVENT = (200+6) # macro
NV_ESC_FREE_OS_EVENT = (200+7) # macro
NV_ESC_STATUS_CODE = (200+9) # macro
NV_ESC_CHECK_VERSION_STR = (200+10) # macro
NV_ESC_IOCTL_XFER_CMD = (200+11) # macro
NV_ESC_ATTACH_GPUS_TO_FD = (200+12) # macro
NV_ESC_QUERY_DEVICE_INTR = (200+13) # macro
NV_ESC_SYS_PARAMS = (200+14) # macro
NV_ESC_EXPORT_TO_DMABUF_FD = (200+17) # macro
NV_ESC_WAIT_OPEN_COMPLETE = (200+18) # macro
NV_RM_API_VERSION_STRING_LENGTH = 64 # macro
NV_RM_API_VERSION_CMD_STRICT = 0 # macro
NV_RM_API_VERSION_CMD_RELAXED = '1' # macro
NV_RM_API_VERSION_CMD_QUERY = '2' # macro
NV_RM_API_VERSION_REPLY_UNRECOGNIZED = 0 # macro
NV_RM_API_VERSION_REPLY_RECOGNIZED = 1 # macro
NV_DMABUF_EXPORT_MAX_HANDLES = 128 # macro
class struct_c__SA_nv_pci_info_t(Structure):
    pass

struct_c__SA_nv_pci_info_t._pack_ = 1 # source:False
struct_c__SA_nv_pci_info_t._fields_ = [
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_ubyte),
    ('slot', ctypes.c_ubyte),
    ('function', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('vendor_id', ctypes.c_uint16),
    ('device_id', ctypes.c_uint16),
]

nv_pci_info_t = struct_c__SA_nv_pci_info_t
class struct_nv_ioctl_xfer(Structure):
    pass

struct_nv_ioctl_xfer._pack_ = 1 # source:False
struct_nv_ioctl_xfer._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('ptr', ctypes.POINTER(None)),
]

nv_ioctl_xfer_t = struct_nv_ioctl_xfer
class struct_nv_ioctl_card_info(Structure):
    pass

struct_nv_ioctl_card_info._pack_ = 1 # source:False
struct_nv_ioctl_card_info._fields_ = [
    ('valid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('pci_info', nv_pci_info_t),
    ('gpu_id', ctypes.c_uint32),
    ('interrupt_line', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('reg_address', ctypes.c_uint64),
    ('reg_size', ctypes.c_uint64),
    ('fb_address', ctypes.c_uint64),
    ('fb_size', ctypes.c_uint64),
    ('minor_number', ctypes.c_uint32),
    ('dev_name', ctypes.c_ubyte * 10),
    ('PADDING_2', ctypes.c_ubyte * 2),
]

nv_ioctl_card_info_t = struct_nv_ioctl_card_info
class struct_nv_ioctl_alloc_os_event(Structure):
    pass

struct_nv_ioctl_alloc_os_event._pack_ = 1 # source:False
struct_nv_ioctl_alloc_os_event._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('fd', ctypes.c_uint32),
    ('Status', ctypes.c_uint32),
]

nv_ioctl_alloc_os_event_t = struct_nv_ioctl_alloc_os_event
class struct_nv_ioctl_free_os_event(Structure):
    pass

struct_nv_ioctl_free_os_event._pack_ = 1 # source:False
struct_nv_ioctl_free_os_event._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('fd', ctypes.c_uint32),
    ('Status', ctypes.c_uint32),
]

nv_ioctl_free_os_event_t = struct_nv_ioctl_free_os_event
class struct_nv_ioctl_status_code(Structure):
    pass

struct_nv_ioctl_status_code._pack_ = 1 # source:False
struct_nv_ioctl_status_code._fields_ = [
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_ubyte),
    ('slot', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
]

nv_ioctl_status_code_t = struct_nv_ioctl_status_code
class struct_nv_ioctl_rm_api_version(Structure):
    pass

struct_nv_ioctl_rm_api_version._pack_ = 1 # source:False
struct_nv_ioctl_rm_api_version._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('reply', ctypes.c_uint32),
    ('versionString', ctypes.c_char * 64),
]

nv_ioctl_rm_api_version_t = struct_nv_ioctl_rm_api_version
class struct_nv_ioctl_query_device_intr(Structure):
    pass

struct_nv_ioctl_query_device_intr._pack_ = 1 # source:False
struct_nv_ioctl_query_device_intr._fields_ = [
    ('intrStatus', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

nv_ioctl_query_device_intr = struct_nv_ioctl_query_device_intr
class struct_nv_ioctl_sys_params(Structure):
    pass

struct_nv_ioctl_sys_params._pack_ = 1 # source:False
struct_nv_ioctl_sys_params._fields_ = [
    ('memblock_size', ctypes.c_uint64),
]

nv_ioctl_sys_params_t = struct_nv_ioctl_sys_params
class struct_nv_ioctl_register_fd(Structure):
    pass

struct_nv_ioctl_register_fd._pack_ = 1 # source:False
struct_nv_ioctl_register_fd._fields_ = [
    ('ctl_fd', ctypes.c_int32),
]

nv_ioctl_register_fd_t = struct_nv_ioctl_register_fd
class struct_nv_ioctl_export_to_dma_buf_fd(Structure):
    pass

struct_nv_ioctl_export_to_dma_buf_fd._pack_ = 1 # source:False
struct_nv_ioctl_export_to_dma_buf_fd._fields_ = [
    ('fd', ctypes.c_int32),
    ('hClient', ctypes.c_uint32),
    ('totalObjects', ctypes.c_uint32),
    ('numObjects', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('totalSize', ctypes.c_uint64),
    ('handles', ctypes.c_uint32 * 128),
    ('offsets', ctypes.c_uint64 * 128),
    ('sizes', ctypes.c_uint64 * 128),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

nv_ioctl_export_to_dma_buf_fd_t = struct_nv_ioctl_export_to_dma_buf_fd
class struct_nv_ioctl_wait_open_complete(Structure):
    pass

struct_nv_ioctl_wait_open_complete._pack_ = 1 # source:False
struct_nv_ioctl_wait_open_complete._fields_ = [
    ('rc', ctypes.c_int32),
    ('adapterStatus', ctypes.c_uint32),
]

nv_ioctl_wait_open_complete_t = struct_nv_ioctl_wait_open_complete
NV_IOCTL_NUMA_H = True # macro
# def __aligned(n):  # macro
#    return __attribute__((aligned(n)))  
NV_ESC_NUMA_INFO = (200+15) # macro
NV_ESC_SET_NUMA_STATUS = (200+16) # macro
NV_IOCTL_NUMA_INFO_MAX_OFFLINE_ADDRESSES = 64 # macro
NV_IOCTL_NUMA_STATUS_DISABLED = 0 # macro
NV_IOCTL_NUMA_STATUS_OFFLINE = 1 # macro
NV_IOCTL_NUMA_STATUS_ONLINE_IN_PROGRESS = 2 # macro
NV_IOCTL_NUMA_STATUS_ONLINE = 3 # macro
NV_IOCTL_NUMA_STATUS_ONLINE_FAILED = 4 # macro
NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS = 5 # macro
NV_IOCTL_NUMA_STATUS_OFFLINE_FAILED = 6 # macro
class struct_offline_addresses(Structure):
    pass

struct_offline_addresses._pack_ = 1 # source:False
struct_offline_addresses._fields_ = [
    ('addresses', ctypes.c_uint64 * 64),
    ('numEntries', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nv_offline_addresses_t = struct_offline_addresses
class struct_nv_ioctl_numa_info(Structure):
    pass

struct_nv_ioctl_numa_info._pack_ = 1 # source:False
struct_nv_ioctl_numa_info._fields_ = [
    ('nid', ctypes.c_int32),
    ('status', ctypes.c_int32),
    ('memblock_size', ctypes.c_uint64),
    ('numa_mem_addr', ctypes.c_uint64),
    ('numa_mem_size', ctypes.c_uint64),
    ('use_auto_online', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('offline_addresses', nv_offline_addresses_t),
]

nv_ioctl_numa_info_t = struct_nv_ioctl_numa_info
class struct_nv_ioctl_set_numa_status(Structure):
    pass

struct_nv_ioctl_set_numa_status._pack_ = 1 # source:False
struct_nv_ioctl_set_numa_status._fields_ = [
    ('status', ctypes.c_int32),
]

nv_ioctl_set_numa_status_t = struct_nv_ioctl_set_numa_status
_NV_UNIX_NVOS_PARAMS_WRAPPERS_H_ = True # macro
NVOS_INCLUDED = True # macro
NVOS04_FLAGS_CHANNEL_TYPE = ['1', ':', '0'] # macro
NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_TYPE_VIRTUAL = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL_FOR_VIRTUAL = 0x00000002 # macro
NVOS04_FLAGS_VPR = ['2', ':', '2'] # macro
NVOS04_FLAGS_VPR_FALSE = 0x00000000 # macro
NVOS04_FLAGS_VPR_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CC_SECURE = ['2', ':', '2'] # macro
NVOS04_FLAGS_CC_SECURE_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CC_SECURE_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING = ['3', ':', '3'] # macro
NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_TRUE = 0x00000001 # macro
NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE = ['4', ':', '4'] # macro
NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_DEFAULT = 0x00000000 # macro
NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE = 0x00000001 # macro
NVOS04_FLAGS_PRIVILEGED_CHANNEL = ['5', ':', '5'] # macro
NVOS04_FLAGS_PRIVILEGED_CHANNEL_FALSE = 0x00000000 # macro
NVOS04_FLAGS_PRIVILEGED_CHANNEL_TRUE = 0x00000001 # macro
NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING = ['6', ':', '6'] # macro
NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_FALSE = 0x00000000 # macro
NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE = ['7', ':', '7'] # macro
NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE = ['10', ':', '8'] # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED = ['11', ':', '11'] # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE = ['20', ':', '12'] # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED = ['21', ':', '21'] # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV = ['22', ':', '22'] # macro
NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER = ['23', ':', '23'] # macro
NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO = ['24', ':', '24'] # macro
NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_TRUE = 0x00000001 # macro
NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL = ['25', ':', '25'] # macro
NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_FALSE = 0x00000000 # macro
NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT = ['26', ':', '26'] # macro
NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_TRUE = 0x00000001 # macro
NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT = ['27', ':', '27'] # macro
NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_FALSE = 0x00000000 # macro
NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_TRUE = 0x00000001 # macro
NVOS04_FLAGS_GROUP_CHANNEL_THREAD = ['29', ':', '28'] # macro
NVOS04_FLAGS_GROUP_CHANNEL_THREAD_DEFAULT = 0x00000000 # macro
NVOS04_FLAGS_GROUP_CHANNEL_THREAD_ONE = 0x00000001 # macro
NVOS04_FLAGS_GROUP_CHANNEL_THREAD_TWO = 0x00000002 # macro
NVOS04_FLAGS_MAP_CHANNEL = ['30', ':', '30'] # macro
NVOS04_FLAGS_MAP_CHANNEL_FALSE = 0x00000000 # macro
NVOS04_FLAGS_MAP_CHANNEL_TRUE = 0x00000001 # macro
NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC = ['31', ':', '31'] # macro
NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_FALSE = 0x00000000 # macro
NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_TRUE = 0x00000001 # macro
CC_CHAN_ALLOC_IV_SIZE_DWORD = 3 # macro
CC_CHAN_ALLOC_NONCE_SIZE_DWORD = 8 # macro
NV_CHANNEL_ALLOC_PARAMS_MESSAGE_ID = (0x906f) # macro
FILE_DEVICE_NV = 0x00008000 # macro
NV_IOCTL_FCT_BASE = 0x00000800 # macro
NVOS_MAX_SUBDEVICES = 8 # macro
UNIFIED_NV_STATUS = 1 # macro
# NVOS_STATUS = NV_STATUS # macro
# NVOS_STATUS_SUCCESS = NV_OK # macro
# NVOS_STATUS_ERROR_CARD_NOT_PRESENT = NV_ERR_CARD_NOT_PRESENT # macro
# NVOS_STATUS_ERROR_DUAL_LINK_INUSE = NV_ERR_DUAL_LINK_INUSE # macro
# NVOS_STATUS_ERROR_GENERIC = NV_ERR_GENERIC # macro
# NVOS_STATUS_ERROR_GPU_NOT_FULL_POWER = NV_ERR_GPU_NOT_FULL_POWER # macro
# NVOS_STATUS_ERROR_ILLEGAL_ACTION = NV_ERR_ILLEGAL_ACTION # macro
# NVOS_STATUS_ERROR_IN_USE = NV_ERR_STATE_IN_USE # macro
# NVOS_STATUS_ERROR_INSUFFICIENT_RESOURCES = NV_ERR_INSUFFICIENT_RESOURCES # macro
# NVOS_STATUS_ERROR_INVALID_ACCESS_TYPE = NV_ERR_INVALID_ACCESS_TYPE # macro
# NVOS_STATUS_ERROR_INVALID_ARGUMENT = NV_ERR_INVALID_ARGUMENT # macro
# NVOS_STATUS_ERROR_INVALID_BASE = NV_ERR_INVALID_BASE # macro
# NVOS_STATUS_ERROR_INVALID_CHANNEL = NV_ERR_INVALID_CHANNEL # macro
# NVOS_STATUS_ERROR_INVALID_CLASS = NV_ERR_INVALID_CLASS # macro
# NVOS_STATUS_ERROR_INVALID_CLIENT = NV_ERR_INVALID_CLIENT # macro
# NVOS_STATUS_ERROR_INVALID_COMMAND = NV_ERR_INVALID_COMMAND # macro
# NVOS_STATUS_ERROR_INVALID_DATA = NV_ERR_INVALID_DATA # macro
# NVOS_STATUS_ERROR_INVALID_DEVICE = NV_ERR_INVALID_DEVICE # macro
# NVOS_STATUS_ERROR_INVALID_DMA_SPECIFIER = NV_ERR_INVALID_DMA_SPECIFIER # macro
# NVOS_STATUS_ERROR_INVALID_EVENT = NV_ERR_INVALID_EVENT # macro
# NVOS_STATUS_ERROR_INVALID_FLAGS = NV_ERR_INVALID_FLAGS # macro
# NVOS_STATUS_ERROR_INVALID_FUNCTION = NV_ERR_INVALID_FUNCTION # macro
# NVOS_STATUS_ERROR_INVALID_HEAP = NV_ERR_INVALID_HEAP # macro
# NVOS_STATUS_ERROR_INVALID_INDEX = NV_ERR_INVALID_INDEX # macro
# NVOS_STATUS_ERROR_INVALID_LIMIT = NV_ERR_INVALID_LIMIT # macro
# NVOS_STATUS_ERROR_INVALID_METHOD = NV_ERR_INVALID_METHOD # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_BUFFER = NV_ERR_BUFFER_TOO_SMALL # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_ERROR = NV_ERR_INVALID_OBJECT # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_HANDLE = NV_ERR_INVALID_OBJECT_HANDLE # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_NEW = NV_ERR_INVALID_OBJECT_NEW # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_OLD = NV_ERR_INVALID_OBJECT_OLD # macro
# NVOS_STATUS_ERROR_INVALID_OBJECT_PARENT = NV_ERR_INVALID_OBJECT_PARENT # macro
# NVOS_STATUS_ERROR_INVALID_OFFSET = NV_ERR_INVALID_OFFSET # macro
# NVOS_STATUS_ERROR_INVALID_OWNER = NV_ERR_INVALID_OWNER # macro
# NVOS_STATUS_ERROR_INVALID_PARAM_STRUCT = NV_ERR_INVALID_PARAM_STRUCT # macro
# NVOS_STATUS_ERROR_INVALID_PARAMETER = NV_ERR_INVALID_PARAMETER # macro
# NVOS_STATUS_ERROR_INVALID_POINTER = NV_ERR_INVALID_POINTER # macro
# NVOS_STATUS_ERROR_INVALID_REGISTRY_KEY = NV_ERR_INVALID_REGISTRY_KEY # macro
# NVOS_STATUS_ERROR_INVALID_STATE = NV_ERR_INVALID_STATE # macro
# NVOS_STATUS_ERROR_INVALID_STRING_LENGTH = NV_ERR_INVALID_STRING_LENGTH # macro
# NVOS_STATUS_ERROR_INVALID_XLATE = NV_ERR_INVALID_XLATE # macro
# NVOS_STATUS_ERROR_IRQ_NOT_FIRING = NV_ERR_IRQ_NOT_FIRING # macro
# NVOS_STATUS_ERROR_MULTIPLE_MEMORY_TYPES = NV_ERR_MULTIPLE_MEMORY_TYPES # macro
# NVOS_STATUS_ERROR_NOT_SUPPORTED = NV_ERR_NOT_SUPPORTED # macro
# NVOS_STATUS_ERROR_OPERATING_SYSTEM = NV_ERR_OPERATING_SYSTEM # macro
# NVOS_STATUS_ERROR_LIB_RM_VERSION_MISMATCH = NV_ERR_LIB_RM_VERSION_MISMATCH # macro
# NVOS_STATUS_ERROR_PROTECTION_FAULT = NV_ERR_PROTECTION_FAULT # macro
# NVOS_STATUS_ERROR_TIMEOUT = NV_ERR_TIMEOUT # macro
# NVOS_STATUS_ERROR_TOO_MANY_PRIMARIES = NV_ERR_TOO_MANY_PRIMARIES # macro
# NVOS_STATUS_ERROR_IRQ_EDGE_TRIGGERED = NV_ERR_IRQ_EDGE_TRIGGERED # macro
# NVOS_STATUS_ERROR_INVALID_OPERATION = NV_ERR_INVALID_OPERATION # macro
# NVOS_STATUS_ERROR_NOT_COMPATIBLE = NV_ERR_NOT_COMPATIBLE # macro
# NVOS_STATUS_ERROR_MORE_PROCESSING_REQUIRED = NV_WARN_MORE_PROCESSING_REQUIRED # macro
# NVOS_STATUS_ERROR_INSUFFICIENT_PERMISSIONS = NV_ERR_INSUFFICIENT_PERMISSIONS # macro
# NVOS_STATUS_ERROR_TIMEOUT_RETRY = NV_ERR_TIMEOUT_RETRY # macro
# NVOS_STATUS_ERROR_NOT_READY = NV_ERR_NOT_READY # macro
# NVOS_STATUS_ERROR_GPU_IS_LOST = NV_ERR_GPU_IS_LOST # macro
# NVOS_STATUS_ERROR_IN_FULLCHIP_RESET = NV_ERR_GPU_IN_FULLCHIP_RESET # macro
# NVOS_STATUS_ERROR_INVALID_LOCK_STATE = NV_ERR_INVALID_LOCK_STATE # macro
# NVOS_STATUS_ERROR_INVALID_ADDRESS = NV_ERR_INVALID_ADDRESS # macro
# NVOS_STATUS_ERROR_INVALID_IRQ_LEVEL = NV_ERR_INVALID_IRQ_LEVEL # macro
# NVOS_STATUS_ERROR_MEMORY_TRAINING_FAILED = NV_ERR_MEMORY_TRAINING_FAILED # macro
# NVOS_STATUS_ERROR_BUSY_RETRY = NV_ERR_BUSY_RETRY # macro
# NVOS_STATUS_ERROR_INSUFFICIENT_POWER = NV_ERR_INSUFFICIENT_POWER # macro
# NVOS_STATUS_ERROR_OBJECT_NOT_FOUND = NV_ERR_OBJECT_NOT_FOUND # macro
# NVOS_STATUS_ERROR_RESOURCE_LOST = NV_ERR_RESOURCE_LOST # macro
# NVOS_STATUS_ERROR_BUFFER_TOO_SMALL = NV_ERR_BUFFER_TOO_SMALL # macro
# NVOS_STATUS_ERROR_RESET_REQUIRED = NV_ERR_RESET_REQUIRED # macro
# NVOS_STATUS_ERROR_INVALID_REQUEST = NV_ERR_INVALID_REQUEST # macro
# NVOS_STATUS_ERROR_PRIV_SEC_VIOLATION = NV_ERR_PRIV_SEC_VIOLATION # macro
# NVOS_STATUS_ERROR_GPU_IN_DEBUG_MODE = NV_ERR_GPU_IN_DEBUG_MODE # macro
# NVOS_STATUS_ERROR_ALREADY_SIGNALLED = NV_ERR_ALREADY_SIGNALLED # macro
NV01_FREE = (0x00000000) # macro
NV01_ROOT = (0x0) # macro
NV01_ROOT_NON_PRIV = (0x00000001) # macro
# NV01_ROOT_USER = NV01_ROOT_CLIENT # macro
NV01_ROOT_CLIENT = (0x00000041) # macro
NV01_ALLOC_MEMORY = (0x00000002) # macro
NVOS02_FLAGS_PHYSICALITY = ['7', ':', '4'] # macro
NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS = (0x00000000) # macro
NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS = (0x00000001) # macro
NVOS02_FLAGS_LOCATION = ['11', ':', '8'] # macro
NVOS02_FLAGS_LOCATION_PCI = (0x00000000) # macro
NVOS02_FLAGS_LOCATION_AGP = (0x00000001) # macro
NVOS02_FLAGS_LOCATION_VIDMEM = (0x00000002) # macro
NVOS02_FLAGS_COHERENCY = ['15', ':', '12'] # macro
NVOS02_FLAGS_COHERENCY_UNCACHED = (0x00000000) # macro
NVOS02_FLAGS_COHERENCY_CACHED = (0x00000001) # macro
NVOS02_FLAGS_COHERENCY_WRITE_COMBINE = (0x00000002) # macro
NVOS02_FLAGS_COHERENCY_WRITE_THROUGH = (0x00000003) # macro
NVOS02_FLAGS_COHERENCY_WRITE_PROTECT = (0x00000004) # macro
NVOS02_FLAGS_COHERENCY_WRITE_BACK = (0x00000005) # macro
NVOS02_FLAGS_ALLOC = ['17', ':', '16'] # macro
NVOS02_FLAGS_ALLOC_NONE = (0x00000001) # macro
NVOS02_FLAGS_GPU_CACHEABLE = ['18', ':', '18'] # macro
NVOS02_FLAGS_GPU_CACHEABLE_NO = (0x00000000) # macro
NVOS02_FLAGS_GPU_CACHEABLE_YES = (0x00000001) # macro
NVOS02_FLAGS_KERNEL_MAPPING = ['19', ':', '19'] # macro
NVOS02_FLAGS_KERNEL_MAPPING_NO_MAP = (0x00000000) # macro
NVOS02_FLAGS_KERNEL_MAPPING_MAP = (0x00000001) # macro
NVOS02_FLAGS_ALLOC_NISO_DISPLAY = ['20', ':', '20'] # macro
NVOS02_FLAGS_ALLOC_NISO_DISPLAY_NO = (0x00000000) # macro
NVOS02_FLAGS_ALLOC_NISO_DISPLAY_YES = (0x00000001) # macro
NVOS02_FLAGS_ALLOC_USER_READ_ONLY = ['21', ':', '21'] # macro
NVOS02_FLAGS_ALLOC_USER_READ_ONLY_NO = (0x00000000) # macro
NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES = (0x00000001) # macro
NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY = ['22', ':', '22'] # macro
NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_NO = (0x00000000) # macro
NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_YES = (0x00000001) # macro
NVOS02_FLAGS_PEER_MAP_OVERRIDE = ['23', ':', '23'] # macro
NVOS02_FLAGS_PEER_MAP_OVERRIDE_DEFAULT = (0x00000000) # macro
NVOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED = (0x00000001) # macro
NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT = ['24', ':', '24'] # macro
NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT_APERTURE = (0x00000001) # macro
NVOS02_FLAGS_MEMORY_PROTECTION = ['26', ':', '25'] # macro
NVOS02_FLAGS_MEMORY_PROTECTION_DEFAULT = (0x00000000) # macro
NVOS02_FLAGS_MEMORY_PROTECTION_PROTECTED = (0x00000001) # macro
NVOS02_FLAGS_MEMORY_PROTECTION_UNPROTECTED = (0x00000002) # macro
NVOS02_FLAGS_MAPPING = ['31', ':', '30'] # macro
NVOS02_FLAGS_MAPPING_DEFAULT = (0x00000000) # macro
NVOS02_FLAGS_MAPPING_NO_MAP = (0x00000001) # macro
NVOS02_FLAGS_MAPPING_NEVER_MAP = (0x00000002) # macro
NVOS03_FLAGS_ACCESS = ['1', ':', '0'] # macro
NVOS03_FLAGS_ACCESS_READ_WRITE = (0x00000000) # macro
NVOS03_FLAGS_ACCESS_READ_ONLY = (0x00000001) # macro
NVOS03_FLAGS_ACCESS_WRITE_ONLY = (0x00000002) # macro
NVOS03_FLAGS_PREALLOCATE = ['2', ':', '2'] # macro
NVOS03_FLAGS_PREALLOCATE_DISABLE = (0x00000000) # macro
NVOS03_FLAGS_PREALLOCATE_ENABLE = (0x00000001) # macro
NVOS03_FLAGS_GPU_MAPPABLE = ['15', ':', '15'] # macro
NVOS03_FLAGS_GPU_MAPPABLE_DISABLE = (0x00000000) # macro
NVOS03_FLAGS_GPU_MAPPABLE_ENABLE = (0x00000001) # macro
NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE = ['16', ':', '16'] # macro
NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_FALSE = (0x00000000) # macro
NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_TRUE = (0x00000001) # macro
NVOS03_FLAGS_PTE_KIND = ['17', ':', '16'] # macro
NVOS03_FLAGS_PTE_KIND_NONE = (0x00000000) # macro
NVOS03_FLAGS_PTE_KIND_BL = (0x00000001) # macro
NVOS03_FLAGS_PTE_KIND_PITCH = (0x00000002) # macro
NVOS03_FLAGS_TYPE = ['23', ':', '20'] # macro
NVOS03_FLAGS_TYPE_NOTIFIER = (0x00000001) # macro
NVOS03_FLAGS_MAPPING = ['20', ':', '20'] # macro
NVOS03_FLAGS_MAPPING_NONE = (0x00000000) # macro
NVOS03_FLAGS_MAPPING_KERNEL = (0x00000001) # macro
NVOS03_FLAGS_CACHE_SNOOP = ['28', ':', '28'] # macro
NVOS03_FLAGS_CACHE_SNOOP_ENABLE = (0x00000000) # macro
NVOS03_FLAGS_CACHE_SNOOP_DISABLE = (0x00000001) # macro
NVOS03_FLAGS_HASH_TABLE = ['29', ':', '29'] # macro
NVOS03_FLAGS_HASH_TABLE_ENABLE = (0x00000000) # macro
NVOS03_FLAGS_HASH_TABLE_DISABLE = (0x00000001) # macro
NV01_ALLOC_OBJECT = (0x00000005) # macro
NV01_EVENT_KERNEL_CALLBACK = (0x00000078) # macro
NV01_EVENT_OS_EVENT = (0x00000079) # macro
NV01_EVENT_WIN32_EVENT = (0x00000079) # macro
NV01_EVENT_KERNEL_CALLBACK_EX = (0x0000007E) # macro
NV01_EVENT_BROADCAST = (0x80000000) # macro
NV01_EVENT_PERMIT_NON_ROOT_EVENT_KERNEL_CALLBACK_CREATION = (0x40000000) # macro
NV01_EVENT_SUBDEVICE_SPECIFIC = (0x20000000) # macro
NV01_EVENT_WITHOUT_EVENT_DATA = (0x10000000) # macro
NV01_EVENT_NONSTALL_INTR = (0x08000000) # macro
NV01_EVENT_CLIENT_RM = (0x04000000) # macro
NV04_I2C_ACCESS = (0x00000013) # macro
NVOS_I2C_ACCESS_MAX_BUFFER_SIZE = 2048 # macro
NVOS20_COMMAND_unused0001 = 0x0001 # macro
NVOS20_COMMAND_unused0002 = 0x0002 # macro
NVOS20_COMMAND_STRING_PRINT = 0x0003 # macro
NV04_ALLOC = (0x00000015) # macro
NVOS64_FLAGS_NONE = (0x00000000) # macro
NVOS64_FLAGS_FINN_SERIALIZED = (0x00000001) # macro
NVOS65_PARAMETERS_VERSION_MAGIC = 0x77FEF81E # macro
NV04_IDLE_CHANNELS = (0x0000001E) # macro
NVOS30_FLAGS_BEHAVIOR = ['3', ':', '0'] # macro
NVOS30_FLAGS_BEHAVIOR_SPIN = (0x00000000) # macro
NVOS30_FLAGS_BEHAVIOR_SLEEP = (0x00000001) # macro
NVOS30_FLAGS_BEHAVIOR_QUERY = (0x00000002) # macro
NVOS30_FLAGS_BEHAVIOR_FORCE_BUSY_CHECK = (0x00000003) # macro
NVOS30_FLAGS_CHANNEL = ['7', ':', '4'] # macro
NVOS30_FLAGS_CHANNEL_LIST = (0x00000000) # macro
NVOS30_FLAGS_CHANNEL_SINGLE = (0x00000001) # macro
NVOS30_FLAGS_IDLE = ['30', ':', '8'] # macro
NVOS30_FLAGS_IDLE_PUSH_BUFFER = (0x00000001) # macro
NVOS30_FLAGS_IDLE_CACHE1 = (0x00000002) # macro
NVOS30_FLAGS_IDLE_GRAPHICS = (0x00000004) # macro
NVOS30_FLAGS_IDLE_MPEG = (0x00000008) # macro
NVOS30_FLAGS_IDLE_MOTION_ESTIMATION = (0x00000010) # macro
NVOS30_FLAGS_IDLE_VIDEO_PROCESSOR = (0x00000020) # macro
NVOS30_FLAGS_IDLE_MSPDEC = (0x00000020) # macro
NVOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR = (0x00000040) # macro
NVOS30_FLAGS_IDLE_MSVLD = (0x00000040) # macro
NVOS30_FLAGS_IDLE_NVDEC0 = (0x00000040) # macro
NVOS30_FLAGS_IDLE_CIPHER_DMA = (0x00000080) # macro
NVOS30_FLAGS_IDLE_SEC = (0x00000080) # macro
NVOS30_FLAGS_IDLE_CALLBACKS = (0x00000100) # macro
NVOS30_FLAGS_IDLE_MSPPP = (0x00000200) # macro
NVOS30_FLAGS_IDLE_CE0 = (0x00000400) # macro
NVOS30_FLAGS_IDLE_CE1 = (0x00000800) # macro
NVOS30_FLAGS_IDLE_CE2 = (0x00001000) # macro
NVOS30_FLAGS_IDLE_CE3 = (0x00002000) # macro
NVOS30_FLAGS_IDLE_CE4 = (0x00004000) # macro
NVOS30_FLAGS_IDLE_CE5 = (0x00008000) # macro
NVOS30_FLAGS_IDLE_VIC = (0x00010000) # macro
NVOS30_FLAGS_IDLE_MSENC = (0x00020000) # macro
NVOS30_FLAGS_IDLE_NVENC0 = (0x00020000) # macro
NVOS30_FLAGS_IDLE_NVENC1 = (0x00040000) # macro
NVOS30_FLAGS_IDLE_NVENC2 = (0x00080000) # macro
NVOS30_FLAGS_IDLE_NVJPG = (0x00100000) # macro
NVOS30_FLAGS_IDLE_NVDEC1 = (0x00200000) # macro
NVOS30_FLAGS_IDLE_NVDEC2 = (0x00400000) # macro
NVOS30_FLAGS_IDLE_ACTIVECHANNELS = (0x00800000) # macro
NVOS30_FLAGS_IDLE_ALL_ENGINES = ((0x00000004)|(0x00000008)|(0x00000010)|(0x00000020)|(0x00000040)|(0x00000080)|(0x00000020)|(0x00000040)|(0x00000080)|(0x00000200)|(0x00000400)|(0x00000800)|(0x00001000)|(0x00002000)|(0x00004000)|(0x00008000)|(0x00020000)|(0x00040000)|(0x00080000)|(0x00010000)|(0x00100000)|(0x00200000)|(0x00400000)) # macro
NVOS30_FLAGS_WAIT_FOR_ELPG_ON = ['31', ':', '31'] # macro
NVOS30_FLAGS_WAIT_FOR_ELPG_ON_NO = (0x00000000) # macro
NVOS30_FLAGS_WAIT_FOR_ELPG_ON_YES = (0x00000001) # macro
NV04_VID_HEAP_CONTROL = (0x00000020) # macro
NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS = 0 # macro
NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY = 1 # macro
NVOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY = 2 # macro
NVOS32_DESCRIPTOR_TYPE_OS_PHYS_ADDR = 3 # macro
NVOS32_DESCRIPTOR_TYPE_OS_FILE_HANDLE = 4 # macro
NVOS32_DESCRIPTOR_TYPE_OS_DMA_BUF_PTR = 5 # macro
NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR = 6 # macro
NVOS32_DESCRIPTOR_TYPE_KERNEL_VIRTUAL_ADDRESS = 7 # macro
NVOS32_FUNCTION_ALLOC_SIZE = 2 # macro
NVOS32_FUNCTION_FREE = 3 # macro
NVOS32_FUNCTION_INFO = 5 # macro
NVOS32_FUNCTION_ALLOC_TILED_PITCH_HEIGHT = 6 # macro
NVOS32_FUNCTION_DUMP = 11 # macro
NVOS32_FUNCTION_ALLOC_SIZE_RANGE = 14 # macro
NVOS32_FUNCTION_REACQUIRE_COMPR = 15 # macro
NVOS32_FUNCTION_RELEASE_COMPR = 16 # macro
NVOS32_FUNCTION_GET_MEM_ALIGNMENT = 18 # macro
NVOS32_FUNCTION_HW_ALLOC = 19 # macro
NVOS32_FUNCTION_HW_FREE = 20 # macro
NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR = 27 # macro
NVOS32_FLAGS_BLOCKINFO_VISIBILITY_CPU = (0x00000001) # macro
NVOS32_IVC_HEAP_NUMBER_DONT_ALLOCATE_ON_IVC_HEAP = 0 # macro
NVAL_MAX_BANKS = (4) # macro
NVAL_MAP_DIRECTION = ['0', ':', '0'] # macro
NVAL_MAP_DIRECTION_DOWN = 0x00000000 # macro
NVAL_MAP_DIRECTION_UP = 0x00000001 # macro
NV_RM_OS32_ALLOC_OS_DESCRIPTOR_WITH_OS32_ATTR = 1 # macro
NVOS32_DELETE_RESOURCES_ALL = 0 # macro
NVOS32_TYPE_IMAGE = 0 # macro
NVOS32_TYPE_DEPTH = 1 # macro
NVOS32_TYPE_TEXTURE = 2 # macro
NVOS32_TYPE_VIDEO = 3 # macro
NVOS32_TYPE_FONT = 4 # macro
NVOS32_TYPE_CURSOR = 5 # macro
NVOS32_TYPE_DMA = 6 # macro
NVOS32_TYPE_INSTANCE = 7 # macro
NVOS32_TYPE_PRIMARY = 8 # macro
NVOS32_TYPE_ZCULL = 9 # macro
NVOS32_TYPE_UNUSED = 10 # macro
NVOS32_TYPE_SHADER_PROGRAM = 11 # macro
NVOS32_TYPE_OWNER_RM = 12 # macro
NVOS32_TYPE_NOTIFIER = 13 # macro
NVOS32_TYPE_RESERVED = 14 # macro
NVOS32_TYPE_PMA = 15 # macro
NVOS32_TYPE_STENCIL = 16 # macro
NVOS32_NUM_MEM_TYPES = 17 # macro
NVOS32_ATTR_NONE = 0x00000000 # macro
NVOS32_ATTR_DEPTH = ['2', ':', '0'] # macro
NVOS32_ATTR_DEPTH_UNKNOWN = 0x00000000 # macro
NVOS32_ATTR_DEPTH_8 = 0x00000001 # macro
NVOS32_ATTR_DEPTH_16 = 0x00000002 # macro
NVOS32_ATTR_DEPTH_24 = 0x00000003 # macro
NVOS32_ATTR_DEPTH_32 = 0x00000004 # macro
NVOS32_ATTR_DEPTH_64 = 0x00000005 # macro
NVOS32_ATTR_DEPTH_128 = 0x00000006 # macro
NVOS32_ATTR_COMPR_COVG = ['3', ':', '3'] # macro
NVOS32_ATTR_COMPR_COVG_DEFAULT = 0x00000000 # macro
NVOS32_ATTR_COMPR_COVG_PROVIDED = 0x00000001 # macro
NVOS32_ATTR_AA_SAMPLES = ['7', ':', '4'] # macro
NVOS32_ATTR_AA_SAMPLES_1 = 0x00000000 # macro
NVOS32_ATTR_AA_SAMPLES_2 = 0x00000001 # macro
NVOS32_ATTR_AA_SAMPLES_4 = 0x00000002 # macro
NVOS32_ATTR_AA_SAMPLES_4_ROTATED = 0x00000003 # macro
NVOS32_ATTR_AA_SAMPLES_6 = 0x00000004 # macro
NVOS32_ATTR_AA_SAMPLES_8 = 0x00000005 # macro
NVOS32_ATTR_AA_SAMPLES_16 = 0x00000006 # macro
NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_8 = 0x00000007 # macro
NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_16 = 0x00000008 # macro
NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_16 = 0x00000009 # macro
NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_32 = 0x0000000A # macro
NVOS32_ATTR_ZCULL = ['11', ':', '10'] # macro
NVOS32_ATTR_ZCULL_NONE = 0x00000000 # macro
NVOS32_ATTR_ZCULL_REQUIRED = 0x00000001 # macro
NVOS32_ATTR_ZCULL_ANY = 0x00000002 # macro
NVOS32_ATTR_ZCULL_SHARED = 0x00000003 # macro
NVOS32_ATTR_COMPR = ['13', ':', '12'] # macro
NVOS32_ATTR_COMPR_NONE = 0x00000000 # macro
NVOS32_ATTR_COMPR_REQUIRED = 0x00000001 # macro
NVOS32_ATTR_COMPR_ANY = 0x00000002 # macro
NVOS32_ATTR_COMPR_PLC_REQUIRED = 0x00000001 # macro
NVOS32_ATTR_COMPR_PLC_ANY = 0x00000002 # macro
NVOS32_ATTR_COMPR_DISABLE_PLC_ANY = 0x00000003 # macro
NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP = ['14', ':', '14'] # macro
NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_NO = 0x00000000 # macro
NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_YES = 0x00000001 # macro
NVOS32_ATTR_FORMAT = ['17', ':', '16'] # macro
NVOS32_ATTR_FORMAT_LOW_FIELD = 16 # macro
NVOS32_ATTR_FORMAT_HIGH_FIELD = 17 # macro
NVOS32_ATTR_FORMAT_PITCH = 0x00000000 # macro
NVOS32_ATTR_FORMAT_SWIZZLED = 0x00000001 # macro
NVOS32_ATTR_FORMAT_BLOCK_LINEAR = 0x00000002 # macro
NVOS32_ATTR_Z_TYPE = ['18', ':', '18'] # macro
NVOS32_ATTR_Z_TYPE_FIXED = 0x00000000 # macro
NVOS32_ATTR_Z_TYPE_FLOAT = 0x00000001 # macro
NVOS32_ATTR_ZS_PACKING = ['21', ':', '19'] # macro
NVOS32_ATTR_ZS_PACKING_S8 = 0x00000000 # macro
NVOS32_ATTR_ZS_PACKING_Z24S8 = 0x00000000 # macro
NVOS32_ATTR_ZS_PACKING_S8Z24 = 0x00000001 # macro
NVOS32_ATTR_ZS_PACKING_Z32 = 0x00000002 # macro
NVOS32_ATTR_ZS_PACKING_Z24X8 = 0x00000003 # macro
NVOS32_ATTR_ZS_PACKING_X8Z24 = 0x00000004 # macro
NVOS32_ATTR_ZS_PACKING_Z32_X24S8 = 0x00000005 # macro
NVOS32_ATTR_ZS_PACKING_X8Z24_X24S8 = 0x00000006 # macro
NVOS32_ATTR_ZS_PACKING_Z16 = 0x00000007 # macro
NVOS32_ATTR_COLOR_PACKING = ['21', ':', '19'] # macro
NVOS32_ATTR_COLOR_PACKING_A8R8G8B8 = 0x00000000 # macro
NVOS32_ATTR_COLOR_PACKING_X8R8G8B8 = 0x00000001 # macro
NVOS32_ATTR_PAGE_SIZE = ['24', ':', '23'] # macro
NVOS32_ATTR_PAGE_SIZE_DEFAULT = 0x00000000 # macro
NVOS32_ATTR_PAGE_SIZE_4KB = 0x00000001 # macro
NVOS32_ATTR_PAGE_SIZE_BIG = 0x00000002 # macro
NVOS32_ATTR_PAGE_SIZE_HUGE = 0x00000003 # macro
NVOS32_ATTR_LOCATION = ['26', ':', '25'] # macro
NVOS32_ATTR_LOCATION_VIDMEM = 0x00000000 # macro
NVOS32_ATTR_LOCATION_PCI = 0x00000001 # macro
NVOS32_ATTR_LOCATION_AGP = 0x00000002 # macro
NVOS32_ATTR_LOCATION_ANY = 0x00000003 # macro
NVOS32_ATTR_PHYSICALITY = ['28', ':', '27'] # macro
NVOS32_ATTR_PHYSICALITY_DEFAULT = 0x00000000 # macro
NVOS32_ATTR_PHYSICALITY_NONCONTIGUOUS = 0x00000001 # macro
NVOS32_ATTR_PHYSICALITY_CONTIGUOUS = 0x00000002 # macro
NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS = 0x00000003 # macro
NVOS32_ATTR_COHERENCY = ['31', ':', '29'] # macro
NVOS32_ATTR_COHERENCY_UNCACHED = 0x00000000 # macro
NVOS32_ATTR_COHERENCY_CACHED = 0x00000001 # macro
NVOS32_ATTR_COHERENCY_WRITE_COMBINE = 0x00000002 # macro
NVOS32_ATTR_COHERENCY_WRITE_THROUGH = 0x00000003 # macro
NVOS32_ATTR_COHERENCY_WRITE_PROTECT = 0x00000004 # macro
NVOS32_ATTR_COHERENCY_WRITE_BACK = 0x00000005 # macro
NVOS32_ATTR2_NONE = 0x00000000 # macro
NVOS32_ATTR2_ZBC = ['1', ':', '0'] # macro
NVOS32_ATTR2_ZBC_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_ZBC_PREFER_NO_ZBC = 0x00000001 # macro
NVOS32_ATTR2_ZBC_PREFER_ZBC = 0x00000002 # macro
NVOS32_ATTR2_ZBC_REQUIRE_ONLY_ZBC = 0x00000003 # macro
NVOS32_ATTR2_ZBC_INVALID = 0x00000003 # macro
NVOS32_ATTR2_GPU_CACHEABLE = ['3', ':', '2'] # macro
NVOS32_ATTR2_GPU_CACHEABLE_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_GPU_CACHEABLE_YES = 0x00000001 # macro
NVOS32_ATTR2_GPU_CACHEABLE_NO = 0x00000002 # macro
NVOS32_ATTR2_GPU_CACHEABLE_INVALID = 0x00000003 # macro
NVOS32_ATTR2_P2P_GPU_CACHEABLE = ['5', ':', '4'] # macro
NVOS32_ATTR2_P2P_GPU_CACHEABLE_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_P2P_GPU_CACHEABLE_YES = 0x00000001 # macro
NVOS32_ATTR2_P2P_GPU_CACHEABLE_NO = 0x00000002 # macro
NVOS32_ATTR2_32BIT_POINTER = ['6', ':', '6'] # macro
NVOS32_ATTR2_32BIT_POINTER_DISABLE = 0x00000000 # macro
NVOS32_ATTR2_32BIT_POINTER_ENABLE = 0x00000001 # macro
NVOS32_ATTR2_FIXED_NUMA_NODE_ID = ['7', ':', '7'] # macro
NVOS32_ATTR2_FIXED_NUMA_NODE_ID_NO = 0x00000000 # macro
NVOS32_ATTR2_FIXED_NUMA_NODE_ID_YES = 0x00000001 # macro
NVOS32_ATTR2_SMMU_ON_GPU = ['10', ':', '8'] # macro
NVOS32_ATTR2_SMMU_ON_GPU_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_SMMU_ON_GPU_DISABLE = 0x00000001 # macro
NVOS32_ATTR2_SMMU_ON_GPU_ENABLE = 0x00000002 # macro
NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN = ['11', ':', '11'] # macro
NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF = 0x0 # macro
NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_ON = 0x1 # macro
NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_DEFAULT = 0x0 # macro
NVOS32_ATTR2_PRIORITY = ['13', ':', '12'] # macro
NVOS32_ATTR2_PRIORITY_DEFAULT = 0x0 # macro
NVOS32_ATTR2_PRIORITY_HIGH = 0x1 # macro
NVOS32_ATTR2_PRIORITY_LOW = 0x2 # macro
NVOS32_ATTR2_INTERNAL = ['14', ':', '14'] # macro
NVOS32_ATTR2_INTERNAL_NO = 0x0 # macro
NVOS32_ATTR2_INTERNAL_YES = 0x1 # macro
NVOS32_ATTR2_PREFER_2C = ['15', ':', '15'] # macro
NVOS32_ATTR2_PREFER_2C_NO = 0x00000000 # macro
NVOS32_ATTR2_PREFER_2C_YES = 0x00000001 # macro
NVOS32_ATTR2_NISO_DISPLAY = ['16', ':', '16'] # macro
NVOS32_ATTR2_NISO_DISPLAY_NO = 0x00000000 # macro
NVOS32_ATTR2_NISO_DISPLAY_YES = 0x00000001 # macro
NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT = ['17', ':', '17'] # macro
NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_NO = 0x00000000 # macro
NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_YES = 0x00000001 # macro
NVOS32_ATTR2_ISO = ['18', ':', '18'] # macro
NVOS32_ATTR2_ISO_NO = 0x00000000 # macro
NVOS32_ATTR2_ISO_YES = 0x00000001 # macro
NVOS32_ATTR2_BLACKLIST = ['19', ':', '19'] # macro
NVOS32_ATTR2_BLACKLIST_ON = 0x00000000 # macro
NVOS32_ATTR2_BLACKLIST_OFF = 0x00000001 # macro
NVOS32_ATTR2_PAGE_OFFLINING = ['19', ':', '19'] # macro
NVOS32_ATTR2_PAGE_OFFLINING_ON = 0x00000000 # macro
NVOS32_ATTR2_PAGE_OFFLINING_OFF = 0x00000001 # macro
NVOS32_ATTR2_PAGE_SIZE_HUGE = ['21', ':', '20'] # macro
NVOS32_ATTR2_PAGE_SIZE_HUGE_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB = 0x00000001 # macro
NVOS32_ATTR2_PAGE_SIZE_HUGE_512MB = 0x00000002 # macro
NVOS32_ATTR2_PROTECTION_USER = ['22', ':', '22'] # macro
NVOS32_ATTR2_PROTECTION_USER_READ_WRITE = 0x00000000 # macro
NVOS32_ATTR2_PROTECTION_USER_READ_ONLY = 0x00000001 # macro
NVOS32_ATTR2_PROTECTION_DEVICE = ['23', ':', '23'] # macro
NVOS32_ATTR2_PROTECTION_DEVICE_READ_WRITE = 0x00000000 # macro
NVOS32_ATTR2_PROTECTION_DEVICE_READ_ONLY = 0x00000001 # macro
NVOS32_ATTR2_USE_EGM = ['24', ':', '24'] # macro
NVOS32_ATTR2_USE_EGM_FALSE = 0x00000000 # macro
NVOS32_ATTR2_USE_EGM_TRUE = 0x00000001 # macro
NVOS32_ATTR2_MEMORY_PROTECTION = ['26', ':', '25'] # macro
NVOS32_ATTR2_MEMORY_PROTECTION_DEFAULT = 0x00000000 # macro
NVOS32_ATTR2_MEMORY_PROTECTION_PROTECTED = 0x00000001 # macro
NVOS32_ATTR2_MEMORY_PROTECTION_UNPROTECTED = 0x00000002 # macro
NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP = ['27', ':', '27'] # macro
NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_NO = 0x00000000 # macro
NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_YES = 0x00000001 # macro
NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM = ['31', ':', '31'] # macro
NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_FALSE = 0x00000000 # macro
NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_TRUE = 0x00000001 # macro
NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT = 0x00000001 # macro
NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_UP = 0x00000002 # macro
NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_DOWN = 0x00000004 # macro
NVOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE = 0x00000008 # macro
NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE = 0x00000010 # macro
NVOS32_ALLOC_FLAGS_BANK_HINT = 0x00000020 # macro
NVOS32_ALLOC_FLAGS_BANK_FORCE = 0x00000040 # macro
NVOS32_ALLOC_FLAGS_ALIGNMENT_HINT = 0x00000080 # macro
NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE = 0x00000100 # macro
NVOS32_ALLOC_FLAGS_BANK_GROW_UP = 0x00000000 # macro
NVOS32_ALLOC_FLAGS_BANK_GROW_DOWN = 0x00000200 # macro
NVOS32_ALLOC_FLAGS_LAZY = 0x00000400 # macro
NVOS32_ALLOC_FLAGS_FORCE_REVERSE_ALLOC = 0x00000800 # macro
NVOS32_ALLOC_FLAGS_NO_SCANOUT = 0x00001000 # macro
NVOS32_ALLOC_FLAGS_PITCH_FORCE = 0x00002000 # macro
NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED = 0x00004000 # macro
NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED = 0x00008000 # macro
NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM = 0x00010000 # macro
NVOS32_ALLOC_FLAGS_USE_BEGIN_END = 0x00020000 # macro
NVOS32_ALLOC_FLAGS_TURBO_CIPHER_ENCRYPTED = 0x00040000 # macro
NVOS32_ALLOC_FLAGS_VIRTUAL = 0x00080000 # macro
NVOS32_ALLOC_FLAGS_FORCE_INTERNAL_INDEX = 0x00100000 # macro
NVOS32_ALLOC_FLAGS_ZCULL_COVG_SPECIFIED = 0x00200000 # macro
NVOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED = 0x00400000 # macro
NVOS32_ALLOC_FLAGS_FORCE_DEDICATED_PDE = 0x00800000 # macro
NVOS32_ALLOC_FLAGS_PROTECTED = 0x01000000 # macro
NVOS32_ALLOC_FLAGS_KERNEL_MAPPING_MAP = 0x02000000 # macro
NVOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE = 0x02000000 # macro
NVOS32_ALLOC_FLAGS_SPARSE = 0x04000000 # macro
NVOS32_ALLOC_FLAGS_USER_READ_ONLY = 0x04000000 # macro
NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY = 0x08000000 # macro
NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED = 0x08000000 # macro
NVOS32_ALLOC_FLAGS_SKIP_RESOURCE_ALLOC = 0x10000000 # macro
NVOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY = 0x20000000 # macro
NVOS32_ALLOC_FLAGS_SKIP_ALIGN_PAD = 0x40000000 # macro
NVOS32_ALLOC_FLAGS_WPR1 = 0x40000000 # macro
NVOS32_ALLOC_FLAGS_ZCULL_DONT_ALLOCATE_SHARED_1X = 0x80000000 # macro
NVOS32_ALLOC_FLAGS_WPR2 = 0x80000000 # macro
NVOS32_ALLOC_INTERNAL_FLAGS_CLIENTALLOC = 0x00000001 # macro
NVOS32_ALLOC_INTERNAL_FLAGS_SKIP_SCRUB = 0x00000004 # macro
NVOS32_ALLOC_FLAGS_MAXIMIZE_4GB_ADDRESS_SPACE = 0x02000000 # macro
NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY = (0x00080000|0x00000400|0x00400000|0x04000000|0x02000000|0x20000000) # macro
NVOS32_ALLOC_COMPR_COVG_SCALE = 10 # macro
NVOS32_ALLOC_COMPR_COVG_BITS = ['1', ':', '0'] # macro
NVOS32_ALLOC_COMPR_COVG_BITS_DEFAULT = 0x00000000 # macro
NVOS32_ALLOC_COMPR_COVG_BITS_1 = 0x00000001 # macro
NVOS32_ALLOC_COMPR_COVG_BITS_2 = 0x00000002 # macro
NVOS32_ALLOC_COMPR_COVG_BITS_4 = 0x00000003 # macro
NVOS32_ALLOC_COMPR_COVG_MAX = ['11', ':', '2'] # macro
NVOS32_ALLOC_COMPR_COVG_MIN = ['21', ':', '12'] # macro
NVOS32_ALLOC_COMPR_COVG_START = ['31', ':', '22'] # macro
NVOS32_ALLOC_ZCULL_COVG_FORMAT = ['3', ':', '0'] # macro
NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_Z = 0x00000000 # macro
NVOS32_ALLOC_ZCULL_COVG_FORMAT_HIGH_RES_Z = 0x00000002 # macro
NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_ZS = 0x00000003 # macro
NVOS32_ALLOC_ZCULL_COVG_FALLBACK = ['4', ':', '4'] # macro
NVOS32_ALLOC_ZCULL_COVG_FALLBACK_DISALLOW = 0x00000000 # macro
NVOS32_ALLOC_ZCULL_COVG_FALLBACK_ALLOW = 0x00000001 # macro
NVOS32_ALLOC_COMPTAG_OFFSET_START = ['19', ':', '0'] # macro
NVOS32_ALLOC_COMPTAG_OFFSET_START_DEFAULT = 0x00000000 # macro
NVOS32_ALLOC_COMPTAG_OFFSET_USAGE = ['31', ':', '30'] # macro
NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_DEFAULT = 0x00000000 # macro
NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_OFF = 0x00000000 # macro
NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_FIXED = 0x00000001 # macro
NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_MIN = 0x00000002 # macro
NVOS32_REALLOC_FLAGS_GROW_ALLOCATION = 0x00000000 # macro
NVOS32_REALLOC_FLAGS_SHRINK_ALLOCATION = 0x00000001 # macro
NVOS32_REALLOC_FLAGS_REALLOC_UP = 0x00000000 # macro
NVOS32_REALLOC_FLAGS_REALLOC_DOWN = 0x00000002 # macro
NVOS32_RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED = 0x000000001 # macro
NVOS32_REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED = 0x000000001 # macro
NVOS32_FREE_FLAGS_MEMORY_HANDLE_PROVIDED = 0x00000001 # macro
NVOS32_DUMP_FLAGS_TYPE = ['1', ':', '0'] # macro
NVOS32_DUMP_FLAGS_TYPE_FB = 0x00000000 # macro
NVOS32_DUMP_FLAGS_TYPE_CLIENT_PD = 0x00000001 # macro
NVOS32_DUMP_FLAGS_TYPE_CLIENT_VA = 0x00000002 # macro
NVOS32_DUMP_FLAGS_TYPE_CLIENT_VAPTE = 0x00000003 # macro
NVOS32_BLOCK_TYPE_FREE = 0xFFFFFFFF # macro
NVOS32_INVALID_BLOCK_FREE_OFFSET = 0xFFFFFFFF # macro
NVOS32_MEM_TAG_NONE = 0x00000000 # macro
NV04_MAP_MEMORY = (0x00000021) # macro
NV04_MAP_MEMORY_FLAGS_NONE = (0x00000000) # macro
NV04_MAP_MEMORY_FLAGS_USER = (0x00004000) # macro
NVOS33_FLAGS_ACCESS = ['1', ':', '0'] # macro
NVOS33_FLAGS_ACCESS_READ_WRITE = (0x00000000) # macro
NVOS33_FLAGS_ACCESS_READ_ONLY = (0x00000001) # macro
NVOS33_FLAGS_ACCESS_WRITE_ONLY = (0x00000002) # macro
NVOS33_FLAGS_PERSISTENT = ['4', ':', '4'] # macro
NVOS33_FLAGS_PERSISTENT_DISABLE = (0x00000000) # macro
NVOS33_FLAGS_PERSISTENT_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_SKIP_SIZE_CHECK = ['8', ':', '8'] # macro
NVOS33_FLAGS_SKIP_SIZE_CHECK_DISABLE = (0x00000000) # macro
NVOS33_FLAGS_SKIP_SIZE_CHECK_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_MEM_SPACE = ['14', ':', '14'] # macro
NVOS33_FLAGS_MEM_SPACE_CLIENT = (0x00000000) # macro
NVOS33_FLAGS_MEM_SPACE_USER = (0x00000001) # macro
NVOS33_FLAGS_MAPPING = ['16', ':', '15'] # macro
NVOS33_FLAGS_MAPPING_DEFAULT = (0x00000000) # macro
NVOS33_FLAGS_MAPPING_DIRECT = (0x00000001) # macro
NVOS33_FLAGS_MAPPING_REFLECTED = (0x00000002) # macro
NVOS33_FLAGS_FIFO_MAPPING = ['17', ':', '17'] # macro
NVOS33_FLAGS_FIFO_MAPPING_DEFAULT = (0x00000000) # macro
NVOS33_FLAGS_FIFO_MAPPING_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_MAP_FIXED = ['18', ':', '18'] # macro
NVOS33_FLAGS_MAP_FIXED_DISABLE = (0x00000000) # macro
NVOS33_FLAGS_MAP_FIXED_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_RESERVE_ON_UNMAP = ['19', ':', '19'] # macro
NVOS33_FLAGS_RESERVE_ON_UNMAP_DISABLE = (0x00000000) # macro
NVOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_OS_DESCRIPTOR = ['22', ':', '22'] # macro
NVOS33_FLAGS_OS_DESCRIPTOR_DISABLE = (0x00000000) # macro
NVOS33_FLAGS_OS_DESCRIPTOR_ENABLE = (0x00000001) # macro
NVOS33_FLAGS_CACHING_TYPE = ['25', ':', '23'] # macro
NVOS33_FLAGS_CACHING_TYPE_CACHED = 0 # macro
NVOS33_FLAGS_CACHING_TYPE_UNCACHED = 1 # macro
NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED = 2 # macro
NVOS33_FLAGS_CACHING_TYPE_WRITEBACK = 5 # macro
NVOS33_FLAGS_CACHING_TYPE_DEFAULT = 6 # macro
NVOS33_FLAGS_CACHING_TYPE_UNCACHED_WEAK = 7 # macro
NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC = ['26', ':', '26'] # macro
NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_NO = (0x00000000) # macro
NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_YES = (0x00000001) # macro
NV04_UNMAP_MEMORY = (0x00000022) # macro
NV04_ACCESS_REGISTRY = (0x00000026) # macro
NVOS38_ACCESS_TYPE_READ_DWORD = 1 # macro
NVOS38_ACCESS_TYPE_WRITE_DWORD = 2 # macro
NVOS38_ACCESS_TYPE_READ_BINARY = 6 # macro
NVOS38_ACCESS_TYPE_WRITE_BINARY = 7 # macro
NVOS38_MAX_REGISTRY_STRING_LENGTH = 256 # macro
NVOS38_MAX_REGISTRY_BINARY_LENGTH = 256 # macro
NV04_ALLOC_CONTEXT_DMA = (0x00000027) # macro
NV04_GET_EVENT_DATA = (0x00000028) # macro
NVSIM01_BUS_XACT = (0x0000002C) # macro
NV04_MAP_MEMORY_DMA = (0x0000002E) # macro
NVOS46_FLAGS_ACCESS = ['1', ':', '0'] # macro
NVOS46_FLAGS_ACCESS_READ_WRITE = (0x00000000) # macro
NVOS46_FLAGS_ACCESS_READ_ONLY = (0x00000001) # macro
NVOS46_FLAGS_ACCESS_WRITE_ONLY = (0x00000002) # macro
NVOS46_FLAGS_32BIT_POINTER = ['2', ':', '2'] # macro
NVOS46_FLAGS_32BIT_POINTER_DISABLE = (0x00000000) # macro
NVOS46_FLAGS_32BIT_POINTER_ENABLE = (0x00000001) # macro
NVOS46_FLAGS_PAGE_KIND = ['3', ':', '3'] # macro
NVOS46_FLAGS_PAGE_KIND_PHYSICAL = (0x00000000) # macro
NVOS46_FLAGS_PAGE_KIND_VIRTUAL = (0x00000001) # macro
NVOS46_FLAGS_CACHE_SNOOP = ['4', ':', '4'] # macro
NVOS46_FLAGS_CACHE_SNOOP_DISABLE = (0x00000000) # macro
NVOS46_FLAGS_CACHE_SNOOP_ENABLE = (0x00000001) # macro
NVOS46_FLAGS_KERNEL_MAPPING = ['5', ':', '5'] # macro
NVOS46_FLAGS_KERNEL_MAPPING_NONE = (0x00000000) # macro
NVOS46_FLAGS_KERNEL_MAPPING_ENABLE = (0x00000001) # macro
NVOS46_FLAGS_SHADER_ACCESS = ['7', ':', '6'] # macro
NVOS46_FLAGS_SHADER_ACCESS_DEFAULT = (0x00000000) # macro
NVOS46_FLAGS_SHADER_ACCESS_READ_ONLY = (0x00000001) # macro
NVOS46_FLAGS_SHADER_ACCESS_WRITE_ONLY = (0x00000002) # macro
NVOS46_FLAGS_SHADER_ACCESS_READ_WRITE = (0x00000003) # macro
NVOS46_FLAGS_PAGE_SIZE = ['11', ':', '8'] # macro
NVOS46_FLAGS_PAGE_SIZE_DEFAULT = (0x00000000) # macro
NVOS46_FLAGS_PAGE_SIZE_4KB = (0x00000001) # macro
NVOS46_FLAGS_PAGE_SIZE_BIG = (0x00000002) # macro
NVOS46_FLAGS_PAGE_SIZE_BOTH = (0x00000003) # macro
NVOS46_FLAGS_PAGE_SIZE_HUGE = (0x00000004) # macro
NVOS46_FLAGS_SYSTEM_L3_ALLOC = ['13', ':', '13'] # macro
NVOS46_FLAGS_SYSTEM_L3_ALLOC_DEFAULT = (0x00000000) # macro
NVOS46_FLAGS_SYSTEM_L3_ALLOC_ENABLE_HINT = (0x00000001) # macro
NVOS46_FLAGS_DMA_OFFSET_GROWS = ['14', ':', '14'] # macro
NVOS46_FLAGS_DMA_OFFSET_GROWS_UP = (0x00000000) # macro
NVOS46_FLAGS_DMA_OFFSET_GROWS_DOWN = (0x00000001) # macro
NVOS46_FLAGS_DMA_OFFSET_FIXED = ['15', ':', '15'] # macro
NVOS46_FLAGS_DMA_OFFSET_FIXED_FALSE = (0x00000000) # macro
NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE = (0x00000001) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP = ['19', ':', '16'] # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_DEFAULT = (0x00000000) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_1 = (0x00000001) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_2 = (0x00000002) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_4 = (0x00000003) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_8 = (0x00000004) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_16 = (0x00000005) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_32 = (0x00000006) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_64 = (0x00000007) # macro
NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_128 = (0x00000008) # macro
NVOS46_FLAGS_P2P = ['27', ':', '20'] # macro
NVOS46_FLAGS_P2P_ENABLE = ['21', ':', '20'] # macro
NVOS46_FLAGS_P2P_ENABLE_NO = (0x00000000) # macro
NVOS46_FLAGS_P2P_ENABLE_YES = (0x00000001) # macro
NVOS46_FLAGS_P2P_ENABLE_NONE = (0x00000000) # macro
NVOS46_FLAGS_P2P_ENABLE_SLI = (0x00000001) # macro
NVOS46_FLAGS_P2P_ENABLE_NOSLI = (0x00000002) # macro
NVOS46_FLAGS_P2P_SUBDEVICE_ID = ['24', ':', '22'] # macro
NVOS46_FLAGS_P2P_SUBDEV_ID_SRC = ['24', ':', '22'] # macro
NVOS46_FLAGS_P2P_SUBDEV_ID_TGT = ['27', ':', '25'] # macro
NVOS46_FLAGS_TLB_LOCK = ['28', ':', '28'] # macro
NVOS46_FLAGS_TLB_LOCK_DISABLE = (0x00000000) # macro
NVOS46_FLAGS_TLB_LOCK_ENABLE = (0x00000001) # macro
NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC = ['29', ':', '29'] # macro
NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_FALSE = (0x00000000) # macro
NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_TRUE = (0x00000001) # macro
NVOS46_FLAGS_DEFER_TLB_INVALIDATION = ['31', ':', '31'] # macro
NVOS46_FLAGS_DEFER_TLB_INVALIDATION_FALSE = (0x00000000) # macro
NVOS46_FLAGS_DEFER_TLB_INVALIDATION_TRUE = (0x00000001) # macro
NV04_UNMAP_MEMORY_DMA = (0x0000002F) # macro
NVOS47_FLAGS_DEFER_TLB_INVALIDATION = ['0', ':', '0'] # macro
NVOS47_FLAGS_DEFER_TLB_INVALIDATION_FALSE = (0x00000000) # macro
NVOS47_FLAGS_DEFER_TLB_INVALIDATION_TRUE = (0x00000001) # macro
NV04_BIND_CONTEXT_DMA = (0x00000031) # macro
NV04_CONTROL = (0x00000036) # macro
NVOS54_FLAGS_NONE = (0x00000000) # macro
NVOS54_FLAGS_IRQL_RAISED = (0x00000001) # macro
NVOS54_FLAGS_LOCK_BYPASS = (0x00000002) # macro
NVOS54_FLAGS_FINN_SERIALIZED = (0x00000004) # macro
NV04_DUP_OBJECT = (0x00000037) # macro
NV04_DUP_HANDLE_FLAGS_NONE = (0x00000000) # macro
NV04_DUP_HANDLE_FLAGS_REJECT_KERNEL_DUP_PRIVILEGE = (0x00000001) # macro
NV04_UPDATE_DEVICE_MAPPING_INFO = (0x00000038) # macro
NV04_SHARE = (0x0000003E) # macro
NV_DEVICE_ALLOCATION_SZNAME_MAXLEN = 128 # macro
NV_DEVICE_ALLOCATION_FLAGS_NONE = (0x00000000) # macro
NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE_GLOBALLY = (0x00000001) # macro
NV_DEVICE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE = (0x00000002) # macro
NV_DEVICE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS = (0x00000004) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SIZE = (0x00000008) # macro
NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE = (0x00000010) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_TARGET = (0x00000020) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SHARED_MANAGEMENT = (0x00000100) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_64k = (0x00000200) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_128k = (0x00000400) # macro
NV_DEVICE_ALLOCATION_FLAGS_RESTRICT_RESERVED_VALIMITS = (0x00000800) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_MIRRORED = (0x00000040) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_PTABLE_PMA_MANAGED = (0x00001000) # macro
NV_DEVICE_ALLOCATION_FLAGS_HOST_VGPU_DEVICE = (0x00002000) # macro
NV_DEVICE_ALLOCATION_FLAGS_PLUGIN_CONTEXT = (0x00004000) # macro
NV_DEVICE_ALLOCATION_FLAGS_VASPACE_REQUIRE_FIXED_OFFSET = (0x00008000) # macro
NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES = (0x00000000) # macro
NV_DEVICE_ALLOCATION_VAMODE_SINGLE_VASPACE = (0x00000001) # macro
NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES = (0x00000002) # macro
NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR = 0x00000000 # macro
NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN = 0x00000001 # macro
NV_CHANNELGPFIFO_NOTIFICATION_TYPE_KEY_ROTATION_STATUS = 0x00000002 # macro
NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1 = 3 # macro
NV_CHANNELGPFIFO_NOTIFICATION_STATUS_VALUE = ['14', ':', '0'] # macro
NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS = ['15', ':', '15'] # macro
NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_TRUE = 0x1 # macro
NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_FALSE = 0x0 # macro
NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB = ['1', ':', '1'] # macro
NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_YES = 0x00000000 # macro
NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_NO = 0x00000001 # macro
NV_SWRUNLIST_QOS_INTR_NONE = 0x00000000 # macro
NV_SWRUNLIST_QOS_INTR_RUNLIST_AND_ENG_IDLE_ENABLE = NVBIT32 ( 0 ) # macro
NV_SWRUNLIST_QOS_INTR_RUNLIST_IDLE_ENABLE = NVBIT32 ( 1 ) # macro
NV_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_ENABLE = NVBIT32 ( 2 ) # macro
NV_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_AND_ENG_IDLE_ENABLE = NVBIT32 ( 3 ) # macro
NV_VP_ALLOCATION_FLAGS_STANDARD_UCODE = (0x00000000) # macro
NV_VP_ALLOCATION_FLAGS_STATIC_UCODE = (0x00000001) # macro
NV_VP_ALLOCATION_FLAGS_DYNAMIC_UCODE = (0x00000002) # macro
NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_VIDEO = (0x00000000) # macro
NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_AUDIO = (0x00000001) # macro
NV04_ADD_VBLANK_CALLBACK = (0x0000003D) # macro
NV_VASPACE_ALLOCATION_FLAGS_NONE = (0x00000000) # macro
NV_VASPACE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE = BIT ( 0 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS = BIT ( 1 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_SHARED_MANAGEMENT = BIT ( 2 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED = BIT ( 3 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_ENABLE_NVLINK_ATS = BIT ( 4 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_IS_MIRRORED = BIT ( 5 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING = BIT ( 6 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_VA_INTERNAL_LIMIT = BIT ( 7 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_ALLOW_ZERO_ADDRESS = BIT ( 8 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_IS_FLA = BIT ( 9 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_SKIP_SCRUB_MEMPOOL = BIT ( 10 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_OPTIMIZE_PTETABLE_MEMPOOL_USAGE = BIT ( 11 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_REQUIRE_FIXED_OFFSET = BIT ( 12 ) # macro
NV_VASPACE_ALLOCATION_FLAGS_PTETABLE_HEAP_MANAGED = BIT ( 13 ) # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_NEW = 0x00 # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_HOST = 0x01 # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_GLOBAL = 0x02 # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_DEVICE = 0x03 # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_FLA = 0x04 # macro
NV_VASPACE_ALLOCATION_INDEX_GPU_MAX = 0x05 # macro
NV_VASPACE_BIG_PAGE_SIZE_64K = (64*1024) # macro
NV_VASPACE_BIG_PAGE_SIZE_128K = (128*1024) # macro
NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT = ['1', ':', '0'] # macro
NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC = (0x00000000) # macro
NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC = (0x00000001) # macro
NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SPECIFIED = (0x00000002) # macro
NV_TIMEOUT_CONTROL_CMD_SET_DEVICE_TIMEOUT = (0x00000002) # macro
NV_TIMEOUT_CONTROL_CMD_RESET_DEVICE_TIMEOUT = (0x00000003) # macro
class struct_NV_MEMORY_DESC_PARAMS(Structure):
    pass

struct_NV_MEMORY_DESC_PARAMS._pack_ = 1 # source:False
struct_NV_MEMORY_DESC_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('addressSpace', ctypes.c_uint32),
    ('cacheAttrib', ctypes.c_uint32),
]

NV_MEMORY_DESC_PARAMS = struct_NV_MEMORY_DESC_PARAMS
class struct_NV_CHANNEL_ALLOC_PARAMS(Structure):
    pass

struct_NV_CHANNEL_ALLOC_PARAMS._pack_ = 1 # source:False
struct_NV_CHANNEL_ALLOC_PARAMS._fields_ = [
    ('hObjectError', ctypes.c_uint32),
    ('hObjectBuffer', ctypes.c_uint32),
    ('gpFifoOffset', ctypes.c_uint64),
    ('gpFifoEntries', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hContextShare', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('hUserdMemory', ctypes.c_uint32 * 8),
    ('userdOffset', ctypes.c_uint64 * 8),
    ('engineType', ctypes.c_uint32),
    ('cid', ctypes.c_uint32),
    ('subDeviceId', ctypes.c_uint32),
    ('hObjectEccError', ctypes.c_uint32),
    ('instanceMem', NV_MEMORY_DESC_PARAMS),
    ('userdMem', NV_MEMORY_DESC_PARAMS),
    ('ramfcMem', NV_MEMORY_DESC_PARAMS),
    ('mthdbufMem', NV_MEMORY_DESC_PARAMS),
    ('hPhysChannelGroup', ctypes.c_uint32),
    ('internalFlags', ctypes.c_uint32),
    ('errorNotifierMem', NV_MEMORY_DESC_PARAMS),
    ('eccErrorNotifierMem', NV_MEMORY_DESC_PARAMS),
    ('ProcessID', ctypes.c_uint32),
    ('SubProcessID', ctypes.c_uint32),
    ('encryptIv', ctypes.c_uint32 * 3),
    ('decryptIv', ctypes.c_uint32 * 3),
    ('hmacNonce', ctypes.c_uint32 * 8),
]

NV_CHANNEL_ALLOC_PARAMS = struct_NV_CHANNEL_ALLOC_PARAMS
NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS = struct_NV_CHANNEL_ALLOC_PARAMS
class struct_c__SA_NVOS00_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS00_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS00_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectOld', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS00_PARAMETERS = struct_c__SA_NVOS00_PARAMETERS
class struct_c__SA_NVOS02_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS02_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS02_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pMemory', ctypes.POINTER(None)),
    ('limit', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS02_PARAMETERS = struct_c__SA_NVOS02_PARAMETERS
class struct_c__SA_NVOS05_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS05_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS05_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS05_PARAMETERS = struct_c__SA_NVOS05_PARAMETERS
Callback1ArgVoidReturn = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))
Callback5ArgVoidReturn = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
class struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK(Structure):
    pass

struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK._pack_ = 1 # source:False
struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK._fields_ = [
    ('func', ctypes.CFUNCTYPE(None, ctypes.POINTER(None))),
    ('arg', ctypes.POINTER(None)),
]

NVOS10_EVENT_KERNEL_CALLBACK = struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK
class struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK_EX(Structure):
    pass

struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK_EX._pack_ = 1 # source:False
struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK_EX._fields_ = [
    ('func', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)),
    ('arg', ctypes.POINTER(None)),
]

NVOS10_EVENT_KERNEL_CALLBACK_EX = struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK_EX
class struct_c__SA_NVOS_I2C_ACCESS_PARAMS(Structure):
    pass

struct_c__SA_NVOS_I2C_ACCESS_PARAMS._pack_ = 1 # source:False
struct_c__SA_NVOS_I2C_ACCESS_PARAMS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('paramSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('paramStructPtr', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS_I2C_ACCESS_PARAMS = struct_c__SA_NVOS_I2C_ACCESS_PARAMS
class struct_c__SA_NVOS21_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS21_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS21_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('pAllocParms', ctypes.POINTER(None)),
    ('paramsSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS21_PARAMETERS = struct_c__SA_NVOS21_PARAMETERS
class struct_c__SA_NVOS64_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS64_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS64_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('pAllocParms', ctypes.POINTER(None)),
    ('pRightsRequested', ctypes.POINTER(None)),
    ('paramsSize', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

NVOS64_PARAMETERS = struct_c__SA_NVOS64_PARAMETERS
class struct_c__SA_NVOS62_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS62_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS62_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('paramSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS62_PARAMETERS = struct_c__SA_NVOS62_PARAMETERS
class struct_c__SA_NVOS65_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS65_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS65_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('paramSize', ctypes.c_uint32),
    ('versionMagic', ctypes.c_uint32),
    ('maskSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS65_PARAMETERS = struct_c__SA_NVOS65_PARAMETERS
class struct_c__SA_NVOS30_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS30_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS30_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('numChannels', ctypes.c_uint32),
    ('phClients', ctypes.POINTER(None)),
    ('phDevices', ctypes.POINTER(None)),
    ('phChannels', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

NVOS30_PARAMETERS = struct_c__SA_NVOS30_PARAMETERS
BindResultFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
class struct_c__SA_NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS._fields_ = [
    ('sgt', ctypes.POINTER(None)),
    ('gem', ctypes.POINTER(None)),
]

NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS = struct_c__SA_NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS
class struct_c__SA_NVOS32_BLOCKINFO(Structure):
    pass

struct_c__SA_NVOS32_BLOCKINFO._pack_ = 1 # source:False
struct_c__SA_NVOS32_BLOCKINFO._fields_ = [
    ('startOffset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

NVOS32_BLOCKINFO = struct_c__SA_NVOS32_BLOCKINFO
class struct_c__SA_NVOS32_PARAMETERS(Structure):
    pass

class union_c__SA_NVOS32_PARAMETERS_data(Union):
    pass

class struct_c__SA_NVOS32_PARAMETERS_0_AllocSize(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_AllocSize._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_AllocSize._fields_ = [
    ('owner', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('attr', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('comprCovg', ctypes.c_uint32),
    ('zcullCovg', ctypes.c_uint32),
    ('partitionStride', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('address', ctypes.POINTER(None)),
    ('rangeBegin', ctypes.c_uint64),
    ('rangeEnd', ctypes.c_uint64),
    ('attr2', ctypes.c_uint32),
    ('ctagOffset', ctypes.c_uint32),
    ('numaNode', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_c__SA_NVOS32_PARAMETERS_0_AllocTiledPitchHeight(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_AllocTiledPitchHeight._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_AllocTiledPitchHeight._fields_ = [
    ('owner', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pitch', ctypes.c_int32),
    ('attr', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('comprCovg', ctypes.c_uint32),
    ('zcullCovg', ctypes.c_uint32),
    ('partitionStride', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('address', ctypes.POINTER(None)),
    ('rangeBegin', ctypes.c_uint64),
    ('rangeEnd', ctypes.c_uint64),
    ('attr2', ctypes.c_uint32),
    ('ctagOffset', ctypes.c_uint32),
    ('numaNode', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_c__SA_NVOS32_PARAMETERS_0_Free(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_Free._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_Free._fields_ = [
    ('owner', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_c__SA_NVOS32_PARAMETERS_0_ReleaseCompr(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_ReleaseCompr._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_ReleaseCompr._fields_ = [
    ('owner', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
]

class struct_c__SA_NVOS32_PARAMETERS_0_ReacquireCompr(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_ReacquireCompr._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_ReacquireCompr._fields_ = [
    ('owner', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
]

class struct_c__SA_NVOS32_PARAMETERS_0_Info(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_Info._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_Info._fields_ = [
    ('attr', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('base', ctypes.c_uint64),
]

class struct_c__SA_NVOS32_PARAMETERS_0_Dump(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_Dump._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_Dump._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pBuffer', ctypes.POINTER(None)),
    ('numBlocks', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_c__SA_NVOS32_PARAMETERS_0_AllocSizeRange(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_AllocSizeRange._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_AllocSizeRange._fields_ = [
    ('owner', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('attr', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('comprCovg', ctypes.c_uint32),
    ('zcullCovg', ctypes.c_uint32),
    ('partitionStride', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('rangeBegin', ctypes.c_uint64),
    ('rangeEnd', ctypes.c_uint64),
    ('address', ctypes.POINTER(None)),
    ('attr2', ctypes.c_uint32),
    ('ctagOffset', ctypes.c_uint32),
    ('numaNode', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_c__SA_NVOS32_PARAMETERS_0_AllocHintAlignment(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_AllocHintAlignment._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_AllocHintAlignment._fields_ = [
    ('alignType', ctypes.c_uint32),
    ('alignAttr', ctypes.c_uint32),
    ('alignInputFlags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('alignSize', ctypes.c_uint64),
    ('alignHeight', ctypes.c_uint32),
    ('alignWidth', ctypes.c_uint32),
    ('alignPitch', ctypes.c_uint32),
    ('alignPad', ctypes.c_uint32),
    ('alignMask', ctypes.c_uint32),
    ('alignOutputFlags', ctypes.c_uint32 * 4),
    ('alignBank', ctypes.c_uint32 * 4),
    ('alignKind', ctypes.c_uint32),
    ('alignAdjust', ctypes.c_uint32),
    ('alignAttr2', ctypes.c_uint32),
]

class struct_c__SA_NVOS32_PARAMETERS_0_HwAlloc(Structure):
    pass

class struct_c__SA_NVOS32_PARAMETERS_0_9_comprInfo(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_9_comprInfo._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_9_comprInfo._fields_ = [
    ('compPageShift', ctypes.c_uint32),
    ('compressedKind', ctypes.c_uint32),
    ('compTagLineMin', ctypes.c_uint32),
    ('compPageIndexLo', ctypes.c_uint32),
    ('compPageIndexHi', ctypes.c_uint32),
    ('compTagLineMultiplier', ctypes.c_uint32),
]

struct_c__SA_NVOS32_PARAMETERS_0_HwAlloc._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_HwAlloc._fields_ = [
    ('allocOwner', ctypes.c_uint32),
    ('allochMemory', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('allocType', ctypes.c_uint32),
    ('allocAttr', ctypes.c_uint32),
    ('allocInputFlags', ctypes.c_uint32),
    ('allocSize', ctypes.c_uint64),
    ('allocHeight', ctypes.c_uint32),
    ('allocWidth', ctypes.c_uint32),
    ('allocPitch', ctypes.c_uint32),
    ('allocMask', ctypes.c_uint32),
    ('allocComprCovg', ctypes.c_uint32),
    ('allocZcullCovg', ctypes.c_uint32),
    ('bindResultFunc', ctypes.POINTER(None)),
    ('pHandle', ctypes.POINTER(None)),
    ('hResourceHandle', ctypes.c_uint32),
    ('retAttr', ctypes.c_uint32),
    ('kind', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('osDeviceHandle', ctypes.c_uint64),
    ('allocAttr2', ctypes.c_uint32),
    ('retAttr2', ctypes.c_uint32),
    ('allocAddr', ctypes.c_uint64),
    ('comprInfo', struct_c__SA_NVOS32_PARAMETERS_0_9_comprInfo),
    ('uncompressedKind', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_c__SA_NVOS32_PARAMETERS_0_HwFree(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_HwFree._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_HwFree._fields_ = [
    ('hResourceHandle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_c__SA_NVOS32_PARAMETERS_0_AllocOsDesc(Structure):
    pass

struct_c__SA_NVOS32_PARAMETERS_0_AllocOsDesc._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS_0_AllocOsDesc._fields_ = [
    ('hMemory', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('attr', ctypes.c_uint32),
    ('attr2', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('descriptor', ctypes.POINTER(None)),
    ('limit', ctypes.c_uint64),
    ('descriptorType', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

union_c__SA_NVOS32_PARAMETERS_data._pack_ = 1 # source:False
union_c__SA_NVOS32_PARAMETERS_data._fields_ = [
    ('AllocSize', struct_c__SA_NVOS32_PARAMETERS_0_AllocSize),
    ('AllocTiledPitchHeight', struct_c__SA_NVOS32_PARAMETERS_0_AllocTiledPitchHeight),
    ('Free', struct_c__SA_NVOS32_PARAMETERS_0_Free),
    ('ReleaseCompr', struct_c__SA_NVOS32_PARAMETERS_0_ReleaseCompr),
    ('ReacquireCompr', struct_c__SA_NVOS32_PARAMETERS_0_ReacquireCompr),
    ('Info', struct_c__SA_NVOS32_PARAMETERS_0_Info),
    ('Dump', struct_c__SA_NVOS32_PARAMETERS_0_Dump),
    ('AllocSizeRange', struct_c__SA_NVOS32_PARAMETERS_0_AllocSizeRange),
    ('AllocHintAlignment', struct_c__SA_NVOS32_PARAMETERS_0_AllocHintAlignment),
    ('HwAlloc', struct_c__SA_NVOS32_PARAMETERS_0_HwAlloc),
    ('HwFree', struct_c__SA_NVOS32_PARAMETERS_0_HwFree),
    ('AllocOsDesc', struct_c__SA_NVOS32_PARAMETERS_0_AllocOsDesc),
    ('PADDING_0', ctypes.c_ubyte * 96),
]

struct_c__SA_NVOS32_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS32_PARAMETERS._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('function', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('ivcHeapNumber', ctypes.c_int16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
    ('total', ctypes.c_uint64),
    ('free', ctypes.c_uint64),
    ('data', union_c__SA_NVOS32_PARAMETERS_data),
]

NVOS32_PARAMETERS = struct_c__SA_NVOS32_PARAMETERS
class struct_c__SA_NVOS32_HEAP_DUMP_BLOCK(Structure):
    pass

struct_c__SA_NVOS32_HEAP_DUMP_BLOCK._pack_ = 1 # source:False
struct_c__SA_NVOS32_HEAP_DUMP_BLOCK._fields_ = [
    ('owner', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('begin', ctypes.c_uint64),
    ('align', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

NVOS32_HEAP_DUMP_BLOCK = struct_c__SA_NVOS32_HEAP_DUMP_BLOCK
class struct_c__SA_NV_CONTEXT_DMA_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_CONTEXT_DMA_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_CONTEXT_DMA_ALLOCATION_PARAMS._fields_ = [
    ('hSubDevice', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
]

NV_CONTEXT_DMA_ALLOCATION_PARAMS = struct_c__SA_NV_CONTEXT_DMA_ALLOCATION_PARAMS
class struct_c__SA_NV_MEMORY_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_MEMORY_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_MEMORY_ALLOCATION_PARAMS._fields_ = [
    ('owner', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pitch', ctypes.c_int32),
    ('attr', ctypes.c_uint32),
    ('attr2', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('comprCovg', ctypes.c_uint32),
    ('zcullCovg', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('rangeLo', ctypes.c_uint64),
    ('rangeHi', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('address', ctypes.POINTER(None)),
    ('ctagOffset', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('internalflags', ctypes.c_uint32),
    ('tag', ctypes.c_uint32),
    ('numaNode', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NV_MEMORY_ALLOCATION_PARAMS = struct_c__SA_NV_MEMORY_ALLOCATION_PARAMS
class struct_c__SA_NV_OS_DESC_MEMORY_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_OS_DESC_MEMORY_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_OS_DESC_MEMORY_ALLOCATION_PARAMS._fields_ = [
    ('type', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('attr', ctypes.c_uint32),
    ('attr2', ctypes.c_uint32),
    ('descriptor', ctypes.POINTER(None)),
    ('limit', ctypes.c_uint64),
    ('descriptorType', ctypes.c_uint32),
    ('tag', ctypes.c_uint32),
]

NV_OS_DESC_MEMORY_ALLOCATION_PARAMS = struct_c__SA_NV_OS_DESC_MEMORY_ALLOCATION_PARAMS
class struct_c__SA_NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('physAddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('tag', ctypes.c_uint32),
    ('bGuestAllocated', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS = struct_c__SA_NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS
class struct_c__SA_NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS._fields_ = [
    ('owner', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('attr', ctypes.c_uint32),
    ('attr2', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
    ('alignment', ctypes.c_uint32),
    ('comprCovg', ctypes.c_uint32),
    ('zcullCovg', ctypes.c_uint32),
    ('kind', ctypes.c_uint32),
    ('bindResultFunc', ctypes.POINTER(None)),
    ('pHandle', ctypes.POINTER(None)),
    ('osDeviceHandle', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('allocAddr', ctypes.c_uint64),
    ('compPageShift', ctypes.c_uint32),
    ('compressedKind', ctypes.c_uint32),
    ('compTagLineMin', ctypes.c_uint32),
    ('compPageIndexLo', ctypes.c_uint32),
    ('compPageIndexHi', ctypes.c_uint32),
    ('compTagLineMultiplier', ctypes.c_uint32),
    ('uncompressedKind', ctypes.c_uint32),
    ('tag', ctypes.c_uint32),
]

NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS = struct_c__SA_NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS
class struct_c__SA_NVOS33_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS33_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS33_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('pLinearAddress', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

NVOS33_PARAMETERS = struct_c__SA_NVOS33_PARAMETERS
class struct_c__SA_NVOS34_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS34_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS34_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pLinearAddress', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

NVOS34_PARAMETERS = struct_c__SA_NVOS34_PARAMETERS
class struct_c__SA_NVOS38_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS38_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS38_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('AccessType', ctypes.c_uint32),
    ('DevNodeLength', ctypes.c_uint32),
    ('pDevNode', ctypes.POINTER(None)),
    ('ParmStrLength', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pParmStr', ctypes.POINTER(None)),
    ('BinaryDataLength', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pBinaryData', ctypes.POINTER(None)),
    ('Data', ctypes.c_uint32),
    ('Entry', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

NVOS38_PARAMETERS = struct_c__SA_NVOS38_PARAMETERS
class struct_c__SA_NVOS39_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS39_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS39_PARAMETERS._fields_ = [
    ('hObjectParent', ctypes.c_uint32),
    ('hSubDevice', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('selector', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS39_PARAMETERS = struct_c__SA_NVOS39_PARAMETERS
class struct_c__SA_NvUnixEvent(Structure):
    pass

struct_c__SA_NvUnixEvent._pack_ = 1 # source:False
struct_c__SA_NvUnixEvent._fields_ = [
    ('hObject', ctypes.c_uint32),
    ('NotifyIndex', ctypes.c_uint32),
    ('info32', ctypes.c_uint32),
    ('info16', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

NvUnixEvent = struct_c__SA_NvUnixEvent
class struct_c__SA_NVOS41_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS41_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS41_PARAMETERS._fields_ = [
    ('pEvent', ctypes.POINTER(None)),
    ('MoreEvents', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS41_PARAMETERS = struct_c__SA_NVOS41_PARAMETERS
class struct_c__SA_NVOS2C_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS2C_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS2C_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('bar', ctypes.c_uint32),
    ('bytes', ctypes.c_uint32),
    ('write', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS2C_PARAMETERS = struct_c__SA_NVOS2C_PARAMETERS
class struct_c__SA_NVOS46_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS46_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS46_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hDma', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('dmaOffset', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS46_PARAMETERS = struct_c__SA_NVOS46_PARAMETERS
class struct_c__SA_NVOS47_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS47_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS47_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hDma', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('dmaOffset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS47_PARAMETERS = struct_c__SA_NVOS47_PARAMETERS
class struct_c__SA_NVOS49_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS49_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS49_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('hCtxDma', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS49_PARAMETERS = struct_c__SA_NVOS49_PARAMETERS
class struct_c__SA_NVOS54_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS54_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS54_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('params', ctypes.POINTER(None)),
    ('paramsSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS54_PARAMETERS = struct_c__SA_NVOS54_PARAMETERS
class struct_c__SA_NVOS63_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS63_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS63_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('paramsSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS63_PARAMETERS = struct_c__SA_NVOS63_PARAMETERS
class struct_c__SA_NVOS55_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS55_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS55_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClientSrc', ctypes.c_uint32),
    ('hObjectSrc', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS55_PARAMETERS = struct_c__SA_NVOS55_PARAMETERS
class struct_c__SA_NVOS56_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS56_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS56_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pOldCpuAddress', ctypes.POINTER(None)),
    ('pNewCpuAddress', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NVOS56_PARAMETERS = struct_c__SA_NVOS56_PARAMETERS
class struct_c__SA_NVOS57_PARAMETERS(Structure):
    pass

class struct_RS_SHARE_POLICY(Structure):
    pass

class struct_RS_ACCESS_MASK(Structure):
    pass

struct_RS_ACCESS_MASK._pack_ = 1 # source:False
struct_RS_ACCESS_MASK._fields_ = [
    ('limbs', ctypes.c_uint32 * 1),
]

struct_RS_SHARE_POLICY._pack_ = 1 # source:False
struct_RS_SHARE_POLICY._fields_ = [
    ('target', ctypes.c_uint32),
    ('accessMask', struct_RS_ACCESS_MASK),
    ('type', ctypes.c_uint16),
    ('action', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

struct_c__SA_NVOS57_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS57_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('sharePolicy', struct_RS_SHARE_POLICY),
    ('status', ctypes.c_uint32),
]

NVOS57_PARAMETERS = struct_c__SA_NVOS57_PARAMETERS
class struct_c__SA_NVPOWERSTATE_PARAMETERS(Structure):
    pass

struct_c__SA_NVPOWERSTATE_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVPOWERSTATE_PARAMETERS._fields_ = [
    ('deviceReference', ctypes.c_uint32),
    ('head', ctypes.c_uint32),
    ('state', ctypes.c_uint32),
    ('forceMonitorState', ctypes.c_ubyte),
    ('bForcePerfBiosLevel', ctypes.c_ubyte),
    ('bIsD3HotTransition', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('fastBootPowerState', ctypes.c_uint32),
]

NVPOWERSTATE_PARAMETERS = struct_c__SA_NVPOWERSTATE_PARAMETERS
PNVPOWERSTATE_PARAMETERS = ctypes.POINTER(struct_c__SA_NVPOWERSTATE_PARAMETERS)
class struct_c__SA_NV_GR_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_GR_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_GR_ALLOCATION_PARAMETERS._fields_ = [
    ('version', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
]

NV_GR_ALLOCATION_PARAMETERS = struct_c__SA_NV_GR_ALLOCATION_PARAMETERS
class struct_c__SA_NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS._fields_ = [
    ('channelInstance', ctypes.c_uint32),
    ('hObjectBuffer', ctypes.c_uint32),
    ('hObjectNotify', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('pControl', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS = struct_c__SA_NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS
class struct_c__SA_NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS._fields_ = [
    ('channelInstance', ctypes.c_uint32),
    ('hObjectNotify', ctypes.c_uint32),
    ('pControl', ctypes.POINTER(None)),
]

NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS = struct_c__SA_NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS
class struct_c__SA_NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS._fields_ = [
    ('hObjectError', ctypes.c_uint32),
    ('hObjectEccError', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('engineType', ctypes.c_uint32),
    ('bIsCallingContextVgpuPlugin', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS = struct_c__SA_NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS
class struct_c__SA_NV_SWRUNLIST_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_SWRUNLIST_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_SWRUNLIST_ALLOCATION_PARAMS._fields_ = [
    ('engineId', ctypes.c_uint32),
    ('maxTSGs', ctypes.c_uint32),
    ('qosIntrEnableMask', ctypes.c_uint32),
]

NV_SWRUNLIST_ALLOCATION_PARAMS = struct_c__SA_NV_SWRUNLIST_ALLOCATION_PARAMS
class struct_c__SA_NV_ME_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_ME_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_ME_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
]

NV_ME_ALLOCATION_PARAMETERS = struct_c__SA_NV_ME_ALLOCATION_PARAMETERS
class struct_c__SA_NV_BSP_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_BSP_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_BSP_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

NV_BSP_ALLOCATION_PARAMETERS = struct_c__SA_NV_BSP_ALLOCATION_PARAMETERS
class struct_c__SA_NV_VP_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_VP_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_VP_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('altUcode', ctypes.c_uint32),
    ('rawUcode', ctypes.POINTER(None)),
    ('rawUcodeSize', ctypes.c_uint32),
    ('numSubClasses', ctypes.c_uint32),
    ('numSubSets', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('subClasses', ctypes.POINTER(None)),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pControl', ctypes.POINTER(None)),
    ('hMemoryCmdBuffer', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
]

NV_VP_ALLOCATION_PARAMETERS = struct_c__SA_NV_VP_ALLOCATION_PARAMETERS
class struct_c__SA_NV_PPP_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_PPP_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_PPP_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
]

NV_PPP_ALLOCATION_PARAMETERS = struct_c__SA_NV_PPP_ALLOCATION_PARAMETERS
class struct_c__SA_NV_MSENC_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_MSENC_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_MSENC_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

NV_MSENC_ALLOCATION_PARAMETERS = struct_c__SA_NV_MSENC_ALLOCATION_PARAMETERS
class struct_c__SA_NV_SEC2_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_SEC2_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_SEC2_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
]

NV_SEC2_ALLOCATION_PARAMETERS = struct_c__SA_NV_SEC2_ALLOCATION_PARAMETERS
class struct_c__SA_NV_NVJPG_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_NVJPG_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_NVJPG_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

NV_NVJPG_ALLOCATION_PARAMETERS = struct_c__SA_NV_NVJPG_ALLOCATION_PARAMETERS
class struct_c__SA_NV_OFA_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_OFA_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_OFA_ALLOCATION_PARAMETERS._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

NV_OFA_ALLOCATION_PARAMETERS = struct_c__SA_NV_OFA_ALLOCATION_PARAMETERS
class struct_c__SA_NVOS61_PARAMETERS(Structure):
    pass

struct_c__SA_NVOS61_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NVOS61_PARAMETERS._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hVblank', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pProc', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None))),
    ('LogicalHead', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pParm1', ctypes.POINTER(None)),
    ('pParm2', ctypes.POINTER(None)),
    ('bAdd', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NVOS61_PARAMETERS = struct_c__SA_NVOS61_PARAMETERS
class struct_c__SA_NV_VASPACE_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_VASPACE_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_VASPACE_ALLOCATION_PARAMETERS._fields_ = [
    ('index', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('vaSize', ctypes.c_uint64),
    ('vaStartInternal', ctypes.c_uint64),
    ('vaLimitInternal', ctypes.c_uint64),
    ('bigPageSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vaBase', ctypes.c_uint64),
]

NV_VASPACE_ALLOCATION_PARAMETERS = struct_c__SA_NV_VASPACE_ALLOCATION_PARAMETERS
class struct_c__SA_NV_CTXSHARE_ALLOCATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_CTXSHARE_ALLOCATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_CTXSHARE_ALLOCATION_PARAMETERS._fields_ = [
    ('hVASpace', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('subctxId', ctypes.c_uint32),
]

NV_CTXSHARE_ALLOCATION_PARAMETERS = struct_c__SA_NV_CTXSHARE_ALLOCATION_PARAMETERS
class struct_c__SA_NV_TIMEOUT_CONTROL_PARAMETERS(Structure):
    pass

struct_c__SA_NV_TIMEOUT_CONTROL_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_TIMEOUT_CONTROL_PARAMETERS._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('timeoutInMs', ctypes.c_uint32),
    ('deviceInstance', ctypes.c_uint32),
]

NV_TIMEOUT_CONTROL_PARAMETERS = struct_c__SA_NV_TIMEOUT_CONTROL_PARAMETERS
class struct_c__SA_NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS(Structure):
    pass

struct_c__SA_NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS._fields_ = [
    ('blockNum', ctypes.c_uint32),
    ('bufferSize', ctypes.c_uint32),
    ('pBuffer', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS = struct_c__SA_NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS
class struct_c__SA_NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS(Structure):
    pass

struct_c__SA_NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS._fields_ = [
    ('bufferSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pBuffer', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS = struct_c__SA_NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS
class struct_c__SA_NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS(Structure):
    pass

struct_c__SA_NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS._pack_ = 1 # source:False
struct_c__SA_NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS._fields_ = [
    ('hParentClient', ctypes.c_uint32),
    ('hSrcResource', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('notifyIndex', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS = struct_c__SA_NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS

# values for enumeration 'c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE'
c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE__enumvalues = {
    0: 'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT',
    1: 'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH',
    2: 'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH',
    3: 'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID',
}
NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT = 0
NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH = 1
NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH = 2
NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID = 3
c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE = ctypes.c_uint32 # enum
NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE = c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE
NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE__enumvalues = c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE__enumvalues
class struct_c__SA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS(Structure):
    pass

struct_c__SA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS._fields_ = [
    ('bDirtyTracking', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('granularity', ctypes.c_uint32),
    ('accessBitMask', ctypes.c_uint64 * 64),
    ('noOfEntries', ctypes.c_uint32),
    ('addrSpace', NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE),
]

NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS = struct_c__SA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS
class struct_c__SA_NV_HOPPER_USERMODE_A_PARAMS(Structure):
    pass

struct_c__SA_NV_HOPPER_USERMODE_A_PARAMS._pack_ = 1 # source:False
struct_c__SA_NV_HOPPER_USERMODE_A_PARAMS._fields_ = [
    ('bBar1Mapping', ctypes.c_ubyte),
    ('bPriv', ctypes.c_ubyte),
]

NV_HOPPER_USERMODE_A_PARAMS = struct_c__SA_NV_HOPPER_USERMODE_A_PARAMS
class struct_c__SA_nv_ioctl_nvos02_parameters_with_fd(Structure):
    pass

struct_c__SA_nv_ioctl_nvos02_parameters_with_fd._pack_ = 1 # source:False
struct_c__SA_nv_ioctl_nvos02_parameters_with_fd._fields_ = [
    ('params', NVOS02_PARAMETERS),
    ('fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nv_ioctl_nvos02_parameters_with_fd = struct_c__SA_nv_ioctl_nvos02_parameters_with_fd
class struct_c__SA_nv_ioctl_nvos33_parameters_with_fd(Structure):
    pass

struct_c__SA_nv_ioctl_nvos33_parameters_with_fd._pack_ = 1 # source:False
struct_c__SA_nv_ioctl_nvos33_parameters_with_fd._fields_ = [
    ('params', NVOS33_PARAMETERS),
    ('fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nv_ioctl_nvos33_parameters_with_fd = struct_c__SA_nv_ioctl_nvos33_parameters_with_fd
__all__ = \
    ['BindResultFunc', 'CC_CHAN_ALLOC_IV_SIZE_DWORD',
    'CC_CHAN_ALLOC_NONCE_SIZE_DWORD', 'Callback1ArgVoidReturn',
    'Callback5ArgVoidReturn', 'FILE_DEVICE_NV', 'NV01_ALLOC_MEMORY',
    'NV01_ALLOC_OBJECT', 'NV01_EVENT_BROADCAST',
    'NV01_EVENT_CLIENT_RM', 'NV01_EVENT_KERNEL_CALLBACK',
    'NV01_EVENT_KERNEL_CALLBACK_EX', 'NV01_EVENT_NONSTALL_INTR',
    'NV01_EVENT_OS_EVENT',
    'NV01_EVENT_PERMIT_NON_ROOT_EVENT_KERNEL_CALLBACK_CREATION',
    'NV01_EVENT_SUBDEVICE_SPECIFIC', 'NV01_EVENT_WIN32_EVENT',
    'NV01_EVENT_WITHOUT_EVENT_DATA', 'NV01_FREE', 'NV01_ROOT',
    'NV01_ROOT_CLIENT', 'NV01_ROOT_NON_PRIV', 'NV04_ACCESS_REGISTRY',
    'NV04_ADD_VBLANK_CALLBACK', 'NV04_ALLOC',
    'NV04_ALLOC_CONTEXT_DMA', 'NV04_BIND_CONTEXT_DMA', 'NV04_CONTROL',
    'NV04_DUP_HANDLE_FLAGS_NONE',
    'NV04_DUP_HANDLE_FLAGS_REJECT_KERNEL_DUP_PRIVILEGE',
    'NV04_DUP_OBJECT', 'NV04_GET_EVENT_DATA', 'NV04_I2C_ACCESS',
    'NV04_IDLE_CHANNELS', 'NV04_MAP_MEMORY', 'NV04_MAP_MEMORY_DMA',
    'NV04_MAP_MEMORY_FLAGS_NONE', 'NV04_MAP_MEMORY_FLAGS_USER',
    'NV04_SHARE', 'NV04_UNMAP_MEMORY', 'NV04_UNMAP_MEMORY_DMA',
    'NV04_UPDATE_DEVICE_MAPPING_INFO', 'NV04_VID_HEAP_CONTROL',
    'NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB',
    'NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_NO',
    'NV50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_YES',
    'NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS',
    'NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS', 'NVAL_MAP_DIRECTION',
    'NVAL_MAP_DIRECTION_DOWN', 'NVAL_MAP_DIRECTION_UP',
    'NVAL_MAX_BANKS', 'NVOS00_PARAMETERS', 'NVOS02_FLAGS_ALLOC',
    'NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY',
    'NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_NO',
    'NVOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_YES',
    'NVOS02_FLAGS_ALLOC_NISO_DISPLAY',
    'NVOS02_FLAGS_ALLOC_NISO_DISPLAY_NO',
    'NVOS02_FLAGS_ALLOC_NISO_DISPLAY_YES', 'NVOS02_FLAGS_ALLOC_NONE',
    'NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT',
    'NVOS02_FLAGS_ALLOC_TYPE_SYNCPOINT_APERTURE',
    'NVOS02_FLAGS_ALLOC_USER_READ_ONLY',
    'NVOS02_FLAGS_ALLOC_USER_READ_ONLY_NO',
    'NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES', 'NVOS02_FLAGS_COHERENCY',
    'NVOS02_FLAGS_COHERENCY_CACHED',
    'NVOS02_FLAGS_COHERENCY_UNCACHED',
    'NVOS02_FLAGS_COHERENCY_WRITE_BACK',
    'NVOS02_FLAGS_COHERENCY_WRITE_COMBINE',
    'NVOS02_FLAGS_COHERENCY_WRITE_PROTECT',
    'NVOS02_FLAGS_COHERENCY_WRITE_THROUGH',
    'NVOS02_FLAGS_GPU_CACHEABLE', 'NVOS02_FLAGS_GPU_CACHEABLE_NO',
    'NVOS02_FLAGS_GPU_CACHEABLE_YES', 'NVOS02_FLAGS_KERNEL_MAPPING',
    'NVOS02_FLAGS_KERNEL_MAPPING_MAP',
    'NVOS02_FLAGS_KERNEL_MAPPING_NO_MAP', 'NVOS02_FLAGS_LOCATION',
    'NVOS02_FLAGS_LOCATION_AGP', 'NVOS02_FLAGS_LOCATION_PCI',
    'NVOS02_FLAGS_LOCATION_VIDMEM', 'NVOS02_FLAGS_MAPPING',
    'NVOS02_FLAGS_MAPPING_DEFAULT', 'NVOS02_FLAGS_MAPPING_NEVER_MAP',
    'NVOS02_FLAGS_MAPPING_NO_MAP', 'NVOS02_FLAGS_MEMORY_PROTECTION',
    'NVOS02_FLAGS_MEMORY_PROTECTION_DEFAULT',
    'NVOS02_FLAGS_MEMORY_PROTECTION_PROTECTED',
    'NVOS02_FLAGS_MEMORY_PROTECTION_UNPROTECTED',
    'NVOS02_FLAGS_PEER_MAP_OVERRIDE',
    'NVOS02_FLAGS_PEER_MAP_OVERRIDE_DEFAULT',
    'NVOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED',
    'NVOS02_FLAGS_PHYSICALITY', 'NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS',
    'NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS', 'NVOS02_PARAMETERS',
    'NVOS03_FLAGS_ACCESS', 'NVOS03_FLAGS_ACCESS_READ_ONLY',
    'NVOS03_FLAGS_ACCESS_READ_WRITE',
    'NVOS03_FLAGS_ACCESS_WRITE_ONLY', 'NVOS03_FLAGS_CACHE_SNOOP',
    'NVOS03_FLAGS_CACHE_SNOOP_DISABLE',
    'NVOS03_FLAGS_CACHE_SNOOP_ENABLE', 'NVOS03_FLAGS_GPU_MAPPABLE',
    'NVOS03_FLAGS_GPU_MAPPABLE_DISABLE',
    'NVOS03_FLAGS_GPU_MAPPABLE_ENABLE', 'NVOS03_FLAGS_HASH_TABLE',
    'NVOS03_FLAGS_HASH_TABLE_DISABLE',
    'NVOS03_FLAGS_HASH_TABLE_ENABLE', 'NVOS03_FLAGS_MAPPING',
    'NVOS03_FLAGS_MAPPING_KERNEL', 'NVOS03_FLAGS_MAPPING_NONE',
    'NVOS03_FLAGS_PREALLOCATE', 'NVOS03_FLAGS_PREALLOCATE_DISABLE',
    'NVOS03_FLAGS_PREALLOCATE_ENABLE', 'NVOS03_FLAGS_PTE_KIND',
    'NVOS03_FLAGS_PTE_KIND_BL', 'NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE',
    'NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_FALSE',
    'NVOS03_FLAGS_PTE_KIND_BL_OVERRIDE_TRUE',
    'NVOS03_FLAGS_PTE_KIND_NONE', 'NVOS03_FLAGS_PTE_KIND_PITCH',
    'NVOS03_FLAGS_TYPE', 'NVOS03_FLAGS_TYPE_NOTIFIER',
    'NVOS04_FLAGS_CC_SECURE', 'NVOS04_FLAGS_CC_SECURE_FALSE',
    'NVOS04_FLAGS_CC_SECURE_TRUE',
    'NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO',
    'NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_FALSE',
    'NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_TRUE',
    'NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV',
    'NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_FALSE',
    'NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_TRUE',
    'NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE',
    'NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_FALSE',
    'NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_TRUE',
    'NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT',
    'NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_FALSE',
    'NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_TRUE',
    'NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING',
    'NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_FALSE',
    'NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_TRUE',
    'NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER',
    'NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_FALSE',
    'NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_TRUE',
    'NVOS04_FLAGS_CHANNEL_TYPE', 'NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL',
    'NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL_FOR_VIRTUAL',
    'NVOS04_FLAGS_CHANNEL_TYPE_VIRTUAL',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_FALSE',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_TRUE',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_FALSE',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_TRUE',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE',
    'NVOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE',
    'NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT',
    'NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_FALSE',
    'NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_TRUE',
    'NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING',
    'NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_FALSE',
    'NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_TRUE',
    'NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE',
    'NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_DEFAULT',
    'NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE',
    'NVOS04_FLAGS_GROUP_CHANNEL_THREAD',
    'NVOS04_FLAGS_GROUP_CHANNEL_THREAD_DEFAULT',
    'NVOS04_FLAGS_GROUP_CHANNEL_THREAD_ONE',
    'NVOS04_FLAGS_GROUP_CHANNEL_THREAD_TWO',
    'NVOS04_FLAGS_MAP_CHANNEL', 'NVOS04_FLAGS_MAP_CHANNEL_FALSE',
    'NVOS04_FLAGS_MAP_CHANNEL_TRUE',
    'NVOS04_FLAGS_PRIVILEGED_CHANNEL',
    'NVOS04_FLAGS_PRIVILEGED_CHANNEL_FALSE',
    'NVOS04_FLAGS_PRIVILEGED_CHANNEL_TRUE',
    'NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL',
    'NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_FALSE',
    'NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_TRUE',
    'NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC',
    'NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_FALSE',
    'NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_TRUE', 'NVOS04_FLAGS_VPR',
    'NVOS04_FLAGS_VPR_FALSE', 'NVOS04_FLAGS_VPR_TRUE',
    'NVOS05_PARAMETERS', 'NVOS10_EVENT_KERNEL_CALLBACK',
    'NVOS10_EVENT_KERNEL_CALLBACK_EX', 'NVOS20_COMMAND_STRING_PRINT',
    'NVOS20_COMMAND_unused0001', 'NVOS20_COMMAND_unused0002',
    'NVOS21_PARAMETERS', 'NVOS2C_PARAMETERS', 'NVOS30_FLAGS_BEHAVIOR',
    'NVOS30_FLAGS_BEHAVIOR_FORCE_BUSY_CHECK',
    'NVOS30_FLAGS_BEHAVIOR_QUERY', 'NVOS30_FLAGS_BEHAVIOR_SLEEP',
    'NVOS30_FLAGS_BEHAVIOR_SPIN', 'NVOS30_FLAGS_CHANNEL',
    'NVOS30_FLAGS_CHANNEL_LIST', 'NVOS30_FLAGS_CHANNEL_SINGLE',
    'NVOS30_FLAGS_IDLE', 'NVOS30_FLAGS_IDLE_ACTIVECHANNELS',
    'NVOS30_FLAGS_IDLE_ALL_ENGINES',
    'NVOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR',
    'NVOS30_FLAGS_IDLE_CACHE1', 'NVOS30_FLAGS_IDLE_CALLBACKS',
    'NVOS30_FLAGS_IDLE_CE0', 'NVOS30_FLAGS_IDLE_CE1',
    'NVOS30_FLAGS_IDLE_CE2', 'NVOS30_FLAGS_IDLE_CE3',
    'NVOS30_FLAGS_IDLE_CE4', 'NVOS30_FLAGS_IDLE_CE5',
    'NVOS30_FLAGS_IDLE_CIPHER_DMA', 'NVOS30_FLAGS_IDLE_GRAPHICS',
    'NVOS30_FLAGS_IDLE_MOTION_ESTIMATION', 'NVOS30_FLAGS_IDLE_MPEG',
    'NVOS30_FLAGS_IDLE_MSENC', 'NVOS30_FLAGS_IDLE_MSPDEC',
    'NVOS30_FLAGS_IDLE_MSPPP', 'NVOS30_FLAGS_IDLE_MSVLD',
    'NVOS30_FLAGS_IDLE_NVDEC0', 'NVOS30_FLAGS_IDLE_NVDEC1',
    'NVOS30_FLAGS_IDLE_NVDEC2', 'NVOS30_FLAGS_IDLE_NVENC0',
    'NVOS30_FLAGS_IDLE_NVENC1', 'NVOS30_FLAGS_IDLE_NVENC2',
    'NVOS30_FLAGS_IDLE_NVJPG', 'NVOS30_FLAGS_IDLE_PUSH_BUFFER',
    'NVOS30_FLAGS_IDLE_SEC', 'NVOS30_FLAGS_IDLE_VIC',
    'NVOS30_FLAGS_IDLE_VIDEO_PROCESSOR',
    'NVOS30_FLAGS_WAIT_FOR_ELPG_ON',
    'NVOS30_FLAGS_WAIT_FOR_ELPG_ON_NO',
    'NVOS30_FLAGS_WAIT_FOR_ELPG_ON_YES', 'NVOS30_PARAMETERS',
    'NVOS32_ALLOC_COMPR_COVG_BITS', 'NVOS32_ALLOC_COMPR_COVG_BITS_1',
    'NVOS32_ALLOC_COMPR_COVG_BITS_2',
    'NVOS32_ALLOC_COMPR_COVG_BITS_4',
    'NVOS32_ALLOC_COMPR_COVG_BITS_DEFAULT',
    'NVOS32_ALLOC_COMPR_COVG_MAX', 'NVOS32_ALLOC_COMPR_COVG_MIN',
    'NVOS32_ALLOC_COMPR_COVG_SCALE', 'NVOS32_ALLOC_COMPR_COVG_START',
    'NVOS32_ALLOC_COMPTAG_OFFSET_START',
    'NVOS32_ALLOC_COMPTAG_OFFSET_START_DEFAULT',
    'NVOS32_ALLOC_COMPTAG_OFFSET_USAGE',
    'NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_DEFAULT',
    'NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_FIXED',
    'NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_MIN',
    'NVOS32_ALLOC_COMPTAG_OFFSET_USAGE_OFF',
    'NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE',
    'NVOS32_ALLOC_FLAGS_ALIGNMENT_HINT',
    'NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED',
    'NVOS32_ALLOC_FLAGS_BANK_FORCE',
    'NVOS32_ALLOC_FLAGS_BANK_GROW_DOWN',
    'NVOS32_ALLOC_FLAGS_BANK_GROW_UP', 'NVOS32_ALLOC_FLAGS_BANK_HINT',
    'NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY',
    'NVOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED',
    'NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE',
    'NVOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE',
    'NVOS32_ALLOC_FLAGS_FORCE_DEDICATED_PDE',
    'NVOS32_ALLOC_FLAGS_FORCE_INTERNAL_INDEX',
    'NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_DOWN',
    'NVOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_UP',
    'NVOS32_ALLOC_FLAGS_FORCE_REVERSE_ALLOC',
    'NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT',
    'NVOS32_ALLOC_FLAGS_KERNEL_MAPPING_MAP',
    'NVOS32_ALLOC_FLAGS_LAZY', 'NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED',
    'NVOS32_ALLOC_FLAGS_MAXIMIZE_4GB_ADDRESS_SPACE',
    'NVOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE',
    'NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED',
    'NVOS32_ALLOC_FLAGS_NO_SCANOUT',
    'NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM',
    'NVOS32_ALLOC_FLAGS_PITCH_FORCE',
    'NVOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY',
    'NVOS32_ALLOC_FLAGS_PROTECTED',
    'NVOS32_ALLOC_FLAGS_SKIP_ALIGN_PAD',
    'NVOS32_ALLOC_FLAGS_SKIP_RESOURCE_ALLOC',
    'NVOS32_ALLOC_FLAGS_SPARSE',
    'NVOS32_ALLOC_FLAGS_TURBO_CIPHER_ENCRYPTED',
    'NVOS32_ALLOC_FLAGS_USER_READ_ONLY',
    'NVOS32_ALLOC_FLAGS_USE_BEGIN_END', 'NVOS32_ALLOC_FLAGS_VIRTUAL',
    'NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY', 'NVOS32_ALLOC_FLAGS_WPR1',
    'NVOS32_ALLOC_FLAGS_WPR2',
    'NVOS32_ALLOC_FLAGS_ZCULL_COVG_SPECIFIED',
    'NVOS32_ALLOC_FLAGS_ZCULL_DONT_ALLOCATE_SHARED_1X',
    'NVOS32_ALLOC_INTERNAL_FLAGS_CLIENTALLOC',
    'NVOS32_ALLOC_INTERNAL_FLAGS_SKIP_SCRUB',
    'NVOS32_ALLOC_ZCULL_COVG_FALLBACK',
    'NVOS32_ALLOC_ZCULL_COVG_FALLBACK_ALLOW',
    'NVOS32_ALLOC_ZCULL_COVG_FALLBACK_DISALLOW',
    'NVOS32_ALLOC_ZCULL_COVG_FORMAT',
    'NVOS32_ALLOC_ZCULL_COVG_FORMAT_HIGH_RES_Z',
    'NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_Z',
    'NVOS32_ALLOC_ZCULL_COVG_FORMAT_LOW_RES_ZS',
    'NVOS32_ATTR2_32BIT_POINTER',
    'NVOS32_ATTR2_32BIT_POINTER_DISABLE',
    'NVOS32_ATTR2_32BIT_POINTER_ENABLE',
    'NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP',
    'NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_NO',
    'NVOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_YES',
    'NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN',
    'NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_DEFAULT',
    'NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF',
    'NVOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_ON',
    'NVOS32_ATTR2_BLACKLIST', 'NVOS32_ATTR2_BLACKLIST_OFF',
    'NVOS32_ATTR2_BLACKLIST_ON', 'NVOS32_ATTR2_FIXED_NUMA_NODE_ID',
    'NVOS32_ATTR2_FIXED_NUMA_NODE_ID_NO',
    'NVOS32_ATTR2_FIXED_NUMA_NODE_ID_YES',
    'NVOS32_ATTR2_GPU_CACHEABLE',
    'NVOS32_ATTR2_GPU_CACHEABLE_DEFAULT',
    'NVOS32_ATTR2_GPU_CACHEABLE_INVALID',
    'NVOS32_ATTR2_GPU_CACHEABLE_NO', 'NVOS32_ATTR2_GPU_CACHEABLE_YES',
    'NVOS32_ATTR2_INTERNAL', 'NVOS32_ATTR2_INTERNAL_NO',
    'NVOS32_ATTR2_INTERNAL_YES', 'NVOS32_ATTR2_ISO',
    'NVOS32_ATTR2_ISO_NO', 'NVOS32_ATTR2_ISO_YES',
    'NVOS32_ATTR2_MEMORY_PROTECTION',
    'NVOS32_ATTR2_MEMORY_PROTECTION_DEFAULT',
    'NVOS32_ATTR2_MEMORY_PROTECTION_PROTECTED',
    'NVOS32_ATTR2_MEMORY_PROTECTION_UNPROTECTED',
    'NVOS32_ATTR2_NISO_DISPLAY', 'NVOS32_ATTR2_NISO_DISPLAY_NO',
    'NVOS32_ATTR2_NISO_DISPLAY_YES', 'NVOS32_ATTR2_NONE',
    'NVOS32_ATTR2_P2P_GPU_CACHEABLE',
    'NVOS32_ATTR2_P2P_GPU_CACHEABLE_DEFAULT',
    'NVOS32_ATTR2_P2P_GPU_CACHEABLE_NO',
    'NVOS32_ATTR2_P2P_GPU_CACHEABLE_YES',
    'NVOS32_ATTR2_PAGE_OFFLINING', 'NVOS32_ATTR2_PAGE_OFFLINING_OFF',
    'NVOS32_ATTR2_PAGE_OFFLINING_ON', 'NVOS32_ATTR2_PAGE_SIZE_HUGE',
    'NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB',
    'NVOS32_ATTR2_PAGE_SIZE_HUGE_512MB',
    'NVOS32_ATTR2_PAGE_SIZE_HUGE_DEFAULT', 'NVOS32_ATTR2_PREFER_2C',
    'NVOS32_ATTR2_PREFER_2C_NO', 'NVOS32_ATTR2_PREFER_2C_YES',
    'NVOS32_ATTR2_PRIORITY', 'NVOS32_ATTR2_PRIORITY_DEFAULT',
    'NVOS32_ATTR2_PRIORITY_HIGH', 'NVOS32_ATTR2_PRIORITY_LOW',
    'NVOS32_ATTR2_PROTECTION_DEVICE',
    'NVOS32_ATTR2_PROTECTION_DEVICE_READ_ONLY',
    'NVOS32_ATTR2_PROTECTION_DEVICE_READ_WRITE',
    'NVOS32_ATTR2_PROTECTION_USER',
    'NVOS32_ATTR2_PROTECTION_USER_READ_ONLY',
    'NVOS32_ATTR2_PROTECTION_USER_READ_WRITE',
    'NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM',
    'NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_FALSE',
    'NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM_TRUE',
    'NVOS32_ATTR2_SMMU_ON_GPU', 'NVOS32_ATTR2_SMMU_ON_GPU_DEFAULT',
    'NVOS32_ATTR2_SMMU_ON_GPU_DISABLE',
    'NVOS32_ATTR2_SMMU_ON_GPU_ENABLE', 'NVOS32_ATTR2_USE_EGM',
    'NVOS32_ATTR2_USE_EGM_FALSE', 'NVOS32_ATTR2_USE_EGM_TRUE',
    'NVOS32_ATTR2_ZBC', 'NVOS32_ATTR2_ZBC_DEFAULT',
    'NVOS32_ATTR2_ZBC_INVALID', 'NVOS32_ATTR2_ZBC_PREFER_NO_ZBC',
    'NVOS32_ATTR2_ZBC_PREFER_ZBC',
    'NVOS32_ATTR2_ZBC_REQUIRE_ONLY_ZBC',
    'NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT',
    'NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_NO',
    'NVOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_YES', 'NVOS32_ATTR_AA_SAMPLES',
    'NVOS32_ATTR_AA_SAMPLES_1', 'NVOS32_ATTR_AA_SAMPLES_16',
    'NVOS32_ATTR_AA_SAMPLES_2', 'NVOS32_ATTR_AA_SAMPLES_4',
    'NVOS32_ATTR_AA_SAMPLES_4_ROTATED',
    'NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_16',
    'NVOS32_ATTR_AA_SAMPLES_4_VIRTUAL_8', 'NVOS32_ATTR_AA_SAMPLES_6',
    'NVOS32_ATTR_AA_SAMPLES_8', 'NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_16',
    'NVOS32_ATTR_AA_SAMPLES_8_VIRTUAL_32',
    'NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP',
    'NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_NO',
    'NVOS32_ATTR_ALLOCATE_FROM_RESERVED_HEAP_YES',
    'NVOS32_ATTR_COHERENCY', 'NVOS32_ATTR_COHERENCY_CACHED',
    'NVOS32_ATTR_COHERENCY_UNCACHED',
    'NVOS32_ATTR_COHERENCY_WRITE_BACK',
    'NVOS32_ATTR_COHERENCY_WRITE_COMBINE',
    'NVOS32_ATTR_COHERENCY_WRITE_PROTECT',
    'NVOS32_ATTR_COHERENCY_WRITE_THROUGH',
    'NVOS32_ATTR_COLOR_PACKING', 'NVOS32_ATTR_COLOR_PACKING_A8R8G8B8',
    'NVOS32_ATTR_COLOR_PACKING_X8R8G8B8', 'NVOS32_ATTR_COMPR',
    'NVOS32_ATTR_COMPR_ANY', 'NVOS32_ATTR_COMPR_COVG',
    'NVOS32_ATTR_COMPR_COVG_DEFAULT',
    'NVOS32_ATTR_COMPR_COVG_PROVIDED',
    'NVOS32_ATTR_COMPR_DISABLE_PLC_ANY', 'NVOS32_ATTR_COMPR_NONE',
    'NVOS32_ATTR_COMPR_PLC_ANY', 'NVOS32_ATTR_COMPR_PLC_REQUIRED',
    'NVOS32_ATTR_COMPR_REQUIRED', 'NVOS32_ATTR_DEPTH',
    'NVOS32_ATTR_DEPTH_128', 'NVOS32_ATTR_DEPTH_16',
    'NVOS32_ATTR_DEPTH_24', 'NVOS32_ATTR_DEPTH_32',
    'NVOS32_ATTR_DEPTH_64', 'NVOS32_ATTR_DEPTH_8',
    'NVOS32_ATTR_DEPTH_UNKNOWN', 'NVOS32_ATTR_FORMAT',
    'NVOS32_ATTR_FORMAT_BLOCK_LINEAR',
    'NVOS32_ATTR_FORMAT_HIGH_FIELD', 'NVOS32_ATTR_FORMAT_LOW_FIELD',
    'NVOS32_ATTR_FORMAT_PITCH', 'NVOS32_ATTR_FORMAT_SWIZZLED',
    'NVOS32_ATTR_LOCATION', 'NVOS32_ATTR_LOCATION_AGP',
    'NVOS32_ATTR_LOCATION_ANY', 'NVOS32_ATTR_LOCATION_PCI',
    'NVOS32_ATTR_LOCATION_VIDMEM', 'NVOS32_ATTR_NONE',
    'NVOS32_ATTR_PAGE_SIZE', 'NVOS32_ATTR_PAGE_SIZE_4KB',
    'NVOS32_ATTR_PAGE_SIZE_BIG', 'NVOS32_ATTR_PAGE_SIZE_DEFAULT',
    'NVOS32_ATTR_PAGE_SIZE_HUGE', 'NVOS32_ATTR_PHYSICALITY',
    'NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS',
    'NVOS32_ATTR_PHYSICALITY_CONTIGUOUS',
    'NVOS32_ATTR_PHYSICALITY_DEFAULT',
    'NVOS32_ATTR_PHYSICALITY_NONCONTIGUOUS', 'NVOS32_ATTR_ZCULL',
    'NVOS32_ATTR_ZCULL_ANY', 'NVOS32_ATTR_ZCULL_NONE',
    'NVOS32_ATTR_ZCULL_REQUIRED', 'NVOS32_ATTR_ZCULL_SHARED',
    'NVOS32_ATTR_ZS_PACKING', 'NVOS32_ATTR_ZS_PACKING_S8',
    'NVOS32_ATTR_ZS_PACKING_S8Z24', 'NVOS32_ATTR_ZS_PACKING_X8Z24',
    'NVOS32_ATTR_ZS_PACKING_X8Z24_X24S8',
    'NVOS32_ATTR_ZS_PACKING_Z16', 'NVOS32_ATTR_ZS_PACKING_Z24S8',
    'NVOS32_ATTR_ZS_PACKING_Z24X8', 'NVOS32_ATTR_ZS_PACKING_Z32',
    'NVOS32_ATTR_ZS_PACKING_Z32_X24S8', 'NVOS32_ATTR_Z_TYPE',
    'NVOS32_ATTR_Z_TYPE_FIXED', 'NVOS32_ATTR_Z_TYPE_FLOAT',
    'NVOS32_BLOCKINFO', 'NVOS32_BLOCK_TYPE_FREE',
    'NVOS32_DELETE_RESOURCES_ALL',
    'NVOS32_DESCRIPTOR_TYPE_KERNEL_VIRTUAL_ADDRESS',
    'NVOS32_DESCRIPTOR_TYPE_OS_DMA_BUF_PTR',
    'NVOS32_DESCRIPTOR_TYPE_OS_FILE_HANDLE',
    'NVOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY',
    'NVOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY',
    'NVOS32_DESCRIPTOR_TYPE_OS_PHYS_ADDR',
    'NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR',
    'NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS',
    'NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS',
    'NVOS32_DUMP_FLAGS_TYPE', 'NVOS32_DUMP_FLAGS_TYPE_CLIENT_PD',
    'NVOS32_DUMP_FLAGS_TYPE_CLIENT_VA',
    'NVOS32_DUMP_FLAGS_TYPE_CLIENT_VAPTE',
    'NVOS32_DUMP_FLAGS_TYPE_FB',
    'NVOS32_FLAGS_BLOCKINFO_VISIBILITY_CPU',
    'NVOS32_FREE_FLAGS_MEMORY_HANDLE_PROVIDED',
    'NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR',
    'NVOS32_FUNCTION_ALLOC_SIZE', 'NVOS32_FUNCTION_ALLOC_SIZE_RANGE',
    'NVOS32_FUNCTION_ALLOC_TILED_PITCH_HEIGHT',
    'NVOS32_FUNCTION_DUMP', 'NVOS32_FUNCTION_FREE',
    'NVOS32_FUNCTION_GET_MEM_ALIGNMENT', 'NVOS32_FUNCTION_HW_ALLOC',
    'NVOS32_FUNCTION_HW_FREE', 'NVOS32_FUNCTION_INFO',
    'NVOS32_FUNCTION_REACQUIRE_COMPR',
    'NVOS32_FUNCTION_RELEASE_COMPR', 'NVOS32_HEAP_DUMP_BLOCK',
    'NVOS32_INVALID_BLOCK_FREE_OFFSET',
    'NVOS32_IVC_HEAP_NUMBER_DONT_ALLOCATE_ON_IVC_HEAP',
    'NVOS32_MEM_TAG_NONE', 'NVOS32_NUM_MEM_TYPES',
    'NVOS32_PARAMETERS',
    'NVOS32_REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED',
    'NVOS32_REALLOC_FLAGS_GROW_ALLOCATION',
    'NVOS32_REALLOC_FLAGS_REALLOC_DOWN',
    'NVOS32_REALLOC_FLAGS_REALLOC_UP',
    'NVOS32_REALLOC_FLAGS_SHRINK_ALLOCATION',
    'NVOS32_RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED',
    'NVOS32_TYPE_CURSOR', 'NVOS32_TYPE_DEPTH', 'NVOS32_TYPE_DMA',
    'NVOS32_TYPE_FONT', 'NVOS32_TYPE_IMAGE', 'NVOS32_TYPE_INSTANCE',
    'NVOS32_TYPE_NOTIFIER', 'NVOS32_TYPE_OWNER_RM', 'NVOS32_TYPE_PMA',
    'NVOS32_TYPE_PRIMARY', 'NVOS32_TYPE_RESERVED',
    'NVOS32_TYPE_SHADER_PROGRAM', 'NVOS32_TYPE_STENCIL',
    'NVOS32_TYPE_TEXTURE', 'NVOS32_TYPE_UNUSED', 'NVOS32_TYPE_VIDEO',
    'NVOS32_TYPE_ZCULL', 'NVOS33_FLAGS_ACCESS',
    'NVOS33_FLAGS_ACCESS_READ_ONLY', 'NVOS33_FLAGS_ACCESS_READ_WRITE',
    'NVOS33_FLAGS_ACCESS_WRITE_ONLY',
    'NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC',
    'NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_NO',
    'NVOS33_FLAGS_ALLOW_MAPPING_ON_HCC_YES',
    'NVOS33_FLAGS_CACHING_TYPE', 'NVOS33_FLAGS_CACHING_TYPE_CACHED',
    'NVOS33_FLAGS_CACHING_TYPE_DEFAULT',
    'NVOS33_FLAGS_CACHING_TYPE_UNCACHED',
    'NVOS33_FLAGS_CACHING_TYPE_UNCACHED_WEAK',
    'NVOS33_FLAGS_CACHING_TYPE_WRITEBACK',
    'NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED',
    'NVOS33_FLAGS_FIFO_MAPPING', 'NVOS33_FLAGS_FIFO_MAPPING_DEFAULT',
    'NVOS33_FLAGS_FIFO_MAPPING_ENABLE', 'NVOS33_FLAGS_MAPPING',
    'NVOS33_FLAGS_MAPPING_DEFAULT', 'NVOS33_FLAGS_MAPPING_DIRECT',
    'NVOS33_FLAGS_MAPPING_REFLECTED', 'NVOS33_FLAGS_MAP_FIXED',
    'NVOS33_FLAGS_MAP_FIXED_DISABLE', 'NVOS33_FLAGS_MAP_FIXED_ENABLE',
    'NVOS33_FLAGS_MEM_SPACE', 'NVOS33_FLAGS_MEM_SPACE_CLIENT',
    'NVOS33_FLAGS_MEM_SPACE_USER', 'NVOS33_FLAGS_OS_DESCRIPTOR',
    'NVOS33_FLAGS_OS_DESCRIPTOR_DISABLE',
    'NVOS33_FLAGS_OS_DESCRIPTOR_ENABLE', 'NVOS33_FLAGS_PERSISTENT',
    'NVOS33_FLAGS_PERSISTENT_DISABLE',
    'NVOS33_FLAGS_PERSISTENT_ENABLE', 'NVOS33_FLAGS_RESERVE_ON_UNMAP',
    'NVOS33_FLAGS_RESERVE_ON_UNMAP_DISABLE',
    'NVOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE',
    'NVOS33_FLAGS_SKIP_SIZE_CHECK',
    'NVOS33_FLAGS_SKIP_SIZE_CHECK_DISABLE',
    'NVOS33_FLAGS_SKIP_SIZE_CHECK_ENABLE', 'NVOS33_PARAMETERS',
    'NVOS34_PARAMETERS', 'NVOS38_ACCESS_TYPE_READ_BINARY',
    'NVOS38_ACCESS_TYPE_READ_DWORD',
    'NVOS38_ACCESS_TYPE_WRITE_BINARY',
    'NVOS38_ACCESS_TYPE_WRITE_DWORD',
    'NVOS38_MAX_REGISTRY_BINARY_LENGTH',
    'NVOS38_MAX_REGISTRY_STRING_LENGTH', 'NVOS38_PARAMETERS',
    'NVOS39_PARAMETERS', 'NVOS41_PARAMETERS',
    'NVOS46_FLAGS_32BIT_POINTER',
    'NVOS46_FLAGS_32BIT_POINTER_DISABLE',
    'NVOS46_FLAGS_32BIT_POINTER_ENABLE', 'NVOS46_FLAGS_ACCESS',
    'NVOS46_FLAGS_ACCESS_READ_ONLY', 'NVOS46_FLAGS_ACCESS_READ_WRITE',
    'NVOS46_FLAGS_ACCESS_WRITE_ONLY', 'NVOS46_FLAGS_CACHE_SNOOP',
    'NVOS46_FLAGS_CACHE_SNOOP_DISABLE',
    'NVOS46_FLAGS_CACHE_SNOOP_ENABLE',
    'NVOS46_FLAGS_DEFER_TLB_INVALIDATION',
    'NVOS46_FLAGS_DEFER_TLB_INVALIDATION_FALSE',
    'NVOS46_FLAGS_DEFER_TLB_INVALIDATION_TRUE',
    'NVOS46_FLAGS_DMA_OFFSET_FIXED',
    'NVOS46_FLAGS_DMA_OFFSET_FIXED_FALSE',
    'NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE',
    'NVOS46_FLAGS_DMA_OFFSET_GROWS',
    'NVOS46_FLAGS_DMA_OFFSET_GROWS_DOWN',
    'NVOS46_FLAGS_DMA_OFFSET_GROWS_UP',
    'NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC',
    'NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_FALSE',
    'NVOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_TRUE',
    'NVOS46_FLAGS_KERNEL_MAPPING',
    'NVOS46_FLAGS_KERNEL_MAPPING_ENABLE',
    'NVOS46_FLAGS_KERNEL_MAPPING_NONE', 'NVOS46_FLAGS_P2P',
    'NVOS46_FLAGS_P2P_ENABLE', 'NVOS46_FLAGS_P2P_ENABLE_NO',
    'NVOS46_FLAGS_P2P_ENABLE_NONE', 'NVOS46_FLAGS_P2P_ENABLE_NOSLI',
    'NVOS46_FLAGS_P2P_ENABLE_SLI', 'NVOS46_FLAGS_P2P_ENABLE_YES',
    'NVOS46_FLAGS_P2P_SUBDEVICE_ID', 'NVOS46_FLAGS_P2P_SUBDEV_ID_SRC',
    'NVOS46_FLAGS_P2P_SUBDEV_ID_TGT', 'NVOS46_FLAGS_PAGE_KIND',
    'NVOS46_FLAGS_PAGE_KIND_PHYSICAL',
    'NVOS46_FLAGS_PAGE_KIND_VIRTUAL', 'NVOS46_FLAGS_PAGE_SIZE',
    'NVOS46_FLAGS_PAGE_SIZE_4KB', 'NVOS46_FLAGS_PAGE_SIZE_BIG',
    'NVOS46_FLAGS_PAGE_SIZE_BOTH', 'NVOS46_FLAGS_PAGE_SIZE_DEFAULT',
    'NVOS46_FLAGS_PAGE_SIZE_HUGE',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_1',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_128',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_16',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_2',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_32',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_4',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_64',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_8',
    'NVOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_DEFAULT',
    'NVOS46_FLAGS_SHADER_ACCESS',
    'NVOS46_FLAGS_SHADER_ACCESS_DEFAULT',
    'NVOS46_FLAGS_SHADER_ACCESS_READ_ONLY',
    'NVOS46_FLAGS_SHADER_ACCESS_READ_WRITE',
    'NVOS46_FLAGS_SHADER_ACCESS_WRITE_ONLY',
    'NVOS46_FLAGS_SYSTEM_L3_ALLOC',
    'NVOS46_FLAGS_SYSTEM_L3_ALLOC_DEFAULT',
    'NVOS46_FLAGS_SYSTEM_L3_ALLOC_ENABLE_HINT',
    'NVOS46_FLAGS_TLB_LOCK', 'NVOS46_FLAGS_TLB_LOCK_DISABLE',
    'NVOS46_FLAGS_TLB_LOCK_ENABLE', 'NVOS46_PARAMETERS',
    'NVOS47_FLAGS_DEFER_TLB_INVALIDATION',
    'NVOS47_FLAGS_DEFER_TLB_INVALIDATION_FALSE',
    'NVOS47_FLAGS_DEFER_TLB_INVALIDATION_TRUE', 'NVOS47_PARAMETERS',
    'NVOS49_PARAMETERS', 'NVOS54_FLAGS_FINN_SERIALIZED',
    'NVOS54_FLAGS_IRQL_RAISED', 'NVOS54_FLAGS_LOCK_BYPASS',
    'NVOS54_FLAGS_NONE', 'NVOS54_PARAMETERS', 'NVOS55_PARAMETERS',
    'NVOS56_PARAMETERS', 'NVOS57_PARAMETERS', 'NVOS61_PARAMETERS',
    'NVOS62_PARAMETERS', 'NVOS63_PARAMETERS',
    'NVOS64_FLAGS_FINN_SERIALIZED', 'NVOS64_FLAGS_NONE',
    'NVOS64_PARAMETERS', 'NVOS65_PARAMETERS',
    'NVOS65_PARAMETERS_VERSION_MAGIC',
    'NVOS_I2C_ACCESS_MAX_BUFFER_SIZE', 'NVOS_I2C_ACCESS_PARAMS',
    'NVOS_INCLUDED', 'NVOS_MAX_SUBDEVICES', 'NVPOWERSTATE_PARAMETERS',
    'NVSIM01_BUS_XACT', 'NV_BSP_ALLOCATION_PARAMETERS',
    'NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS',
    'NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS',
    'NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_FALSE',
    'NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_TRUE',
    'NV_CHANNELGPFIFO_NOTIFICATION_STATUS_VALUE',
    'NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR',
    'NV_CHANNELGPFIFO_NOTIFICATION_TYPE_KEY_ROTATION_STATUS',
    'NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN',
    'NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1',
    'NV_CHANNEL_ALLOC_PARAMS', 'NV_CHANNEL_ALLOC_PARAMS_MESSAGE_ID',
    'NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS',
    'NV_CONTEXT_DMA_ALLOCATION_PARAMS',
    'NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT',
    'NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC',
    'NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SPECIFIED',
    'NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC',
    'NV_CTXSHARE_ALLOCATION_PARAMETERS',
    'NV_DEVICE_ALLOCATION_FLAGS_HOST_VGPU_DEVICE',
    'NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE',
    'NV_DEVICE_ALLOCATION_FLAGS_MAP_PTE_GLOBALLY',
    'NV_DEVICE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE',
    'NV_DEVICE_ALLOCATION_FLAGS_NONE',
    'NV_DEVICE_ALLOCATION_FLAGS_PLUGIN_CONTEXT',
    'NV_DEVICE_ALLOCATION_FLAGS_RESTRICT_RESERVED_VALIMITS',
    'NV_DEVICE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_128k',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_64k',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_MIRRORED',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_TARGET',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_PTABLE_PMA_MANAGED',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_REQUIRE_FIXED_OFFSET',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SHARED_MANAGEMENT',
    'NV_DEVICE_ALLOCATION_FLAGS_VASPACE_SIZE',
    'NV_DEVICE_ALLOCATION_SZNAME_MAXLEN',
    'NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES',
    'NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES',
    'NV_DEVICE_ALLOCATION_VAMODE_SINGLE_VASPACE',
    'NV_DMABUF_EXPORT_MAX_HANDLES', 'NV_ESCAPE_H_INCLUDED',
    'NV_ESC_ALLOC_OS_EVENT', 'NV_ESC_ATTACH_GPUS_TO_FD',
    'NV_ESC_CARD_INFO', 'NV_ESC_CHECK_VERSION_STR',
    'NV_ESC_EXPORT_TO_DMABUF_FD', 'NV_ESC_FREE_OS_EVENT',
    'NV_ESC_IOCTL_XFER_CMD', 'NV_ESC_NUMA_INFO',
    'NV_ESC_QUERY_DEVICE_INTR', 'NV_ESC_REGISTER_FD',
    'NV_ESC_RM_ACCESS_REGISTRY', 'NV_ESC_RM_ADD_VBLANK_CALLBACK',
    'NV_ESC_RM_ALLOC', 'NV_ESC_RM_ALLOC_CONTEXT_DMA2',
    'NV_ESC_RM_ALLOC_MEMORY', 'NV_ESC_RM_ALLOC_OBJECT',
    'NV_ESC_RM_BIND_CONTEXT_DMA', 'NV_ESC_RM_CONFIG_GET',
    'NV_ESC_RM_CONFIG_GET_EX', 'NV_ESC_RM_CONFIG_SET',
    'NV_ESC_RM_CONFIG_SET_EX', 'NV_ESC_RM_CONTROL',
    'NV_ESC_RM_DUP_OBJECT', 'NV_ESC_RM_EXPORT_OBJECT_TO_FD',
    'NV_ESC_RM_FREE', 'NV_ESC_RM_GET_EVENT_DATA',
    'NV_ESC_RM_I2C_ACCESS', 'NV_ESC_RM_IDLE_CHANNELS',
    'NV_ESC_RM_IMPORT_OBJECT_FROM_FD',
    'NV_ESC_RM_LOCKLESS_DIAGNOSTIC', 'NV_ESC_RM_MAP_MEMORY',
    'NV_ESC_RM_MAP_MEMORY_DMA', 'NV_ESC_RM_SHARE',
    'NV_ESC_RM_UNMAP_MEMORY', 'NV_ESC_RM_UNMAP_MEMORY_DMA',
    'NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO',
    'NV_ESC_RM_VID_HEAP_CONTROL', 'NV_ESC_SET_NUMA_STATUS',
    'NV_ESC_STATUS_CODE', 'NV_ESC_SYS_PARAMS',
    'NV_ESC_WAIT_OPEN_COMPLETE', 'NV_GR_ALLOCATION_PARAMETERS',
    'NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS',
    'NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS',
    'NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS',
    'NV_HOPPER_USERMODE_A_PARAMS', 'NV_IOCTL_BASE',
    'NV_IOCTL_FCT_BASE', 'NV_IOCTL_H', 'NV_IOCTL_MAGIC',
    'NV_IOCTL_NUMA_H', 'NV_IOCTL_NUMA_INFO_MAX_OFFLINE_ADDRESSES',
    'NV_IOCTL_NUMA_STATUS_DISABLED', 'NV_IOCTL_NUMA_STATUS_OFFLINE',
    'NV_IOCTL_NUMA_STATUS_OFFLINE_FAILED',
    'NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS',
    'NV_IOCTL_NUMA_STATUS_ONLINE',
    'NV_IOCTL_NUMA_STATUS_ONLINE_FAILED',
    'NV_IOCTL_NUMA_STATUS_ONLINE_IN_PROGRESS', 'NV_IOCTL_NUMBERS_H',
    'NV_MEMORY_ALLOCATION_PARAMS', 'NV_MEMORY_DESC_PARAMS',
    'NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS',
    'NV_ME_ALLOCATION_PARAMETERS', 'NV_MSENC_ALLOCATION_PARAMETERS',
    'NV_NVJPG_ALLOCATION_PARAMETERS', 'NV_OFA_ALLOCATION_PARAMETERS',
    'NV_OS_DESC_MEMORY_ALLOCATION_PARAMS',
    'NV_PPP_ALLOCATION_PARAMETERS', 'NV_RM_API_VERSION_CMD_QUERY',
    'NV_RM_API_VERSION_CMD_RELAXED', 'NV_RM_API_VERSION_CMD_STRICT',
    'NV_RM_API_VERSION_REPLY_RECOGNIZED',
    'NV_RM_API_VERSION_REPLY_UNRECOGNIZED',
    'NV_RM_API_VERSION_STRING_LENGTH',
    'NV_RM_OS32_ALLOC_OS_DESCRIPTOR_WITH_OS32_ATTR',
    'NV_SEC2_ALLOCATION_PARAMETERS', 'NV_SWRUNLIST_ALLOCATION_PARAMS',
    'NV_SWRUNLIST_QOS_INTR_NONE',
    'NV_TIMEOUT_CONTROL_CMD_RESET_DEVICE_TIMEOUT',
    'NV_TIMEOUT_CONTROL_CMD_SET_DEVICE_TIMEOUT',
    'NV_TIMEOUT_CONTROL_PARAMETERS',
    'NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS',
    'NV_VASPACE_ALLOCATION_FLAGS_NONE',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_DEVICE',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_FLA',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_GLOBAL',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_HOST',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_MAX',
    'NV_VASPACE_ALLOCATION_INDEX_GPU_NEW',
    'NV_VASPACE_ALLOCATION_PARAMETERS',
    'NV_VASPACE_BIG_PAGE_SIZE_128K', 'NV_VASPACE_BIG_PAGE_SIZE_64K',
    'NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS',
    'NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE',
    'NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE__enumvalues',
    'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH',
    'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT',
    'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH',
    'NV_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID',
    'NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_AUDIO',
    'NV_VP_ALLOCATION_FLAGS_AVP_CLIENT_VIDEO',
    'NV_VP_ALLOCATION_FLAGS_DYNAMIC_UCODE',
    'NV_VP_ALLOCATION_FLAGS_STANDARD_UCODE',
    'NV_VP_ALLOCATION_FLAGS_STATIC_UCODE',
    'NV_VP_ALLOCATION_PARAMETERS', 'NvUnixEvent',
    'PNVPOWERSTATE_PARAMETERS', 'UNIFIED_NV_STATUS',
    '_NV_UNIX_NVOS_PARAMS_WRAPPERS_H_',
    'c__EA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE',
    'nv_ioctl_alloc_os_event_t', 'nv_ioctl_card_info_t',
    'nv_ioctl_export_to_dma_buf_fd_t', 'nv_ioctl_free_os_event_t',
    'nv_ioctl_numa_info_t', 'nv_ioctl_nvos02_parameters_with_fd',
    'nv_ioctl_nvos33_parameters_with_fd',
    'nv_ioctl_query_device_intr', 'nv_ioctl_register_fd_t',
    'nv_ioctl_rm_api_version_t', 'nv_ioctl_set_numa_status_t',
    'nv_ioctl_status_code_t', 'nv_ioctl_sys_params_t',
    'nv_ioctl_wait_open_complete_t', 'nv_ioctl_xfer_t',
    'nv_offline_addresses_t', 'nv_pci_info_t',
    'struct_NV_CHANNEL_ALLOC_PARAMS', 'struct_NV_MEMORY_DESC_PARAMS',
    'struct_RS_ACCESS_MASK', 'struct_RS_SHARE_POLICY',
    'struct_c__SA_NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS',
    'struct_c__SA_NVOS00_PARAMETERS',
    'struct_c__SA_NVOS02_PARAMETERS',
    'struct_c__SA_NVOS05_PARAMETERS',
    'struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK',
    'struct_c__SA_NVOS10_EVENT_KERNEL_CALLBACK_EX',
    'struct_c__SA_NVOS21_PARAMETERS',
    'struct_c__SA_NVOS2C_PARAMETERS',
    'struct_c__SA_NVOS30_PARAMETERS', 'struct_c__SA_NVOS32_BLOCKINFO',
    'struct_c__SA_NVOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS',
    'struct_c__SA_NVOS32_HEAP_DUMP_BLOCK',
    'struct_c__SA_NVOS32_PARAMETERS',
    'struct_c__SA_NVOS32_PARAMETERS_0_9_comprInfo',
    'struct_c__SA_NVOS32_PARAMETERS_0_AllocHintAlignment',
    'struct_c__SA_NVOS32_PARAMETERS_0_AllocOsDesc',
    'struct_c__SA_NVOS32_PARAMETERS_0_AllocSize',
    'struct_c__SA_NVOS32_PARAMETERS_0_AllocSizeRange',
    'struct_c__SA_NVOS32_PARAMETERS_0_AllocTiledPitchHeight',
    'struct_c__SA_NVOS32_PARAMETERS_0_Dump',
    'struct_c__SA_NVOS32_PARAMETERS_0_Free',
    'struct_c__SA_NVOS32_PARAMETERS_0_HwAlloc',
    'struct_c__SA_NVOS32_PARAMETERS_0_HwFree',
    'struct_c__SA_NVOS32_PARAMETERS_0_Info',
    'struct_c__SA_NVOS32_PARAMETERS_0_ReacquireCompr',
    'struct_c__SA_NVOS32_PARAMETERS_0_ReleaseCompr',
    'struct_c__SA_NVOS33_PARAMETERS',
    'struct_c__SA_NVOS34_PARAMETERS',
    'struct_c__SA_NVOS38_PARAMETERS',
    'struct_c__SA_NVOS39_PARAMETERS',
    'struct_c__SA_NVOS41_PARAMETERS',
    'struct_c__SA_NVOS46_PARAMETERS',
    'struct_c__SA_NVOS47_PARAMETERS',
    'struct_c__SA_NVOS49_PARAMETERS',
    'struct_c__SA_NVOS54_PARAMETERS',
    'struct_c__SA_NVOS55_PARAMETERS',
    'struct_c__SA_NVOS56_PARAMETERS',
    'struct_c__SA_NVOS57_PARAMETERS',
    'struct_c__SA_NVOS61_PARAMETERS',
    'struct_c__SA_NVOS62_PARAMETERS',
    'struct_c__SA_NVOS63_PARAMETERS',
    'struct_c__SA_NVOS64_PARAMETERS',
    'struct_c__SA_NVOS65_PARAMETERS',
    'struct_c__SA_NVOS_I2C_ACCESS_PARAMS',
    'struct_c__SA_NVPOWERSTATE_PARAMETERS',
    'struct_c__SA_NV_BSP_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_CONTEXT_DMA_ALLOCATION_PARAMS',
    'struct_c__SA_NV_CTXSHARE_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_GR_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_GSP_TEST_GET_MSG_BLOCK_PARAMETERS',
    'struct_c__SA_NV_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS',
    'struct_c__SA_NV_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS',
    'struct_c__SA_NV_HOPPER_USERMODE_A_PARAMS',
    'struct_c__SA_NV_MEMORY_ALLOCATION_PARAMS',
    'struct_c__SA_NV_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS',
    'struct_c__SA_NV_ME_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_MSENC_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_NVJPG_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_OFA_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_OS_DESC_MEMORY_ALLOCATION_PARAMS',
    'struct_c__SA_NV_PPP_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_SEC2_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_SWRUNLIST_ALLOCATION_PARAMS',
    'struct_c__SA_NV_TIMEOUT_CONTROL_PARAMETERS',
    'struct_c__SA_NV_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS',
    'struct_c__SA_NV_VASPACE_ALLOCATION_PARAMETERS',
    'struct_c__SA_NV_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS',
    'struct_c__SA_NV_VP_ALLOCATION_PARAMETERS',
    'struct_c__SA_NvUnixEvent',
    'struct_c__SA_nv_ioctl_nvos02_parameters_with_fd',
    'struct_c__SA_nv_ioctl_nvos33_parameters_with_fd',
    'struct_c__SA_nv_pci_info_t', 'struct_nv_ioctl_alloc_os_event',
    'struct_nv_ioctl_card_info',
    'struct_nv_ioctl_export_to_dma_buf_fd',
    'struct_nv_ioctl_free_os_event', 'struct_nv_ioctl_numa_info',
    'struct_nv_ioctl_query_device_intr',
    'struct_nv_ioctl_register_fd', 'struct_nv_ioctl_rm_api_version',
    'struct_nv_ioctl_set_numa_status', 'struct_nv_ioctl_status_code',
    'struct_nv_ioctl_sys_params',
    'struct_nv_ioctl_wait_open_complete', 'struct_nv_ioctl_xfer',
    'struct_offline_addresses', 'union_c__SA_NVOS32_PARAMETERS_data']
