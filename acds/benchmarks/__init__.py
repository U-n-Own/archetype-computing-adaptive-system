from .adiac.getter import get_adiac_data
from .mackey_glass import get_mackey_glass, get_mackey_glass_windows
from .mnist import get_mnist_data
from .mallat import get_mallat_data
from .trace import get_trace_data
from .libras import get_libras_data
from .memory_capacity import get_memory_capacity
from .pathx import get_pathx_data
from .ucr import get_ucr_data, get_olive_oil_data, get_forda_data, get_fordb_data

__all__ = [
	"get_adiac_data",
	"get_mackey_glass",
	"get_mnist_data",
	"get_mackey_glass_windows",
	"get_mallat_data",
	"get_trace_data",
	"get_libras_data",
	"get_memory_capacity",
	"get_pathx_data",
	# UCR generic and convenience wrappers
	"get_ucr_data",
	"get_olive_oil_data",
	"get_forda_data",
	"get_fordb_data",
]
