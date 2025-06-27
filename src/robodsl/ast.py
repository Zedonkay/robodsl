"""Abstract Syntax Tree for RoboDSL.

This module defines the AST nodes for the RoboDSL language, providing
a clean representation of parsed configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Union
from enum import Enum


class QoSReliability(Enum):
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class QoSDurability(Enum):
    TRANSIENT_LOCAL = "transient_local"
    VOLATILE = "volatile"


class QoSHistory(Enum):
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"


class QoSLiveliness(Enum):
    AUTOMATIC = "automatic"
    MANUAL_BY_TOPIC = "manual_by_topic"


@dataclass
class QoSConfig:
    """Quality of Service configuration."""
    reliability: Optional[QoSReliability] = None
    durability: Optional[QoSDurability] = None
    history: Optional[QoSHistory] = None
    depth: int = 10
    deadline: Optional[int] = None
    lifespan: Optional[int] = None
    liveliness: Optional[QoSLiveliness] = None
    liveliness_lease_duration: Optional[int] = None


@dataclass
class PublisherConfig:
    """Publisher configuration."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10


@dataclass
class SubscriberConfig:
    """Subscriber configuration."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10


@dataclass
class ServiceConfig:
    """Service configuration."""
    service: str
    srv_type: str
    qos: Optional[QoSConfig] = None


@dataclass
class ActionConfig:
    """Action configuration."""
    name: str
    action_type: str
    qos: Optional[QoSConfig] = None


@dataclass
class ParameterConfig:
    """Parameter configuration."""
    name: str
    type: str
    default: Any
    description: str = ""
    read_only: bool = False


@dataclass
class LifecycleConfig:
    """Lifecycle configuration."""
    enabled: bool = False
    autostart: bool = True
    cleanup_on_shutdown: bool = True


@dataclass
class TimerConfig:
    """Timer configuration."""
    period: float
    callback: str
    oneshot: bool = False
    autostart: bool = True


@dataclass
class RemapRule:
    """Topic remapping rule."""
    from_topic: str
    to_topic: str


@dataclass
class NodeConfig:
    """Node configuration."""
    name: str
    publishers: List[PublisherConfig] = field(default_factory=list)
    subscribers: List[SubscriberConfig] = field(default_factory=list)
    services: List[ServiceConfig] = field(default_factory=list)
    actions: List[ActionConfig] = field(default_factory=list)
    parameters: List[ParameterConfig] = field(default_factory=list)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    timers: List[TimerConfig] = field(default_factory=list)
    namespace: str = ""
    remap: List[RemapRule] = field(default_factory=list)
    parameter_callbacks: bool = False


class KernelParameterDirection(Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"


@dataclass
class KernelParameter:
    """CUDA kernel parameter."""
    name: str
    type: str
    direction: KernelParameterDirection
    is_const: bool = False
    is_pointer: bool = False
    size_expr: Optional[str] = None


@dataclass
class CudaKernelConfig:
    """CUDA kernel configuration."""
    name: str
    parameters: List[KernelParameter] = field(default_factory=list)
    block_size: tuple = (256, 1, 1)
    grid_size: Optional[tuple] = None
    shared_mem_bytes: int = 0
    use_thrust: bool = False
    code: str = ""
    includes: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)


@dataclass
class RoboDSLConfig:
    """Top-level RoboDSL configuration."""
    project_name: str = "robodsl_project"
    nodes: List[NodeConfig] = field(default_factory=list)
    cuda_kernels: List[CudaKernelConfig] = field(default_factory=list)
    includes: set[str] = field(default_factory=set)


# AST Node classes for the parse tree
@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class IncludeNode(ASTNode):
    """Include statement node."""
    path: str
    is_system: bool  # True for <>, False for ""


@dataclass
class ValueNode(ASTNode):
    """Value node."""
    value: Any


@dataclass
class ParameterNode(ASTNode):
    """Parameter node."""
    name: str
    value: ValueNode


@dataclass
class LifecycleSettingNode(ASTNode):
    """Lifecycle setting node."""
    name: str
    value: bool


@dataclass
class LifecycleNode(ASTNode):
    """Lifecycle configuration node."""
    settings: List[LifecycleSettingNode]


@dataclass
class TimerSettingNode(ASTNode):
    """Timer setting node."""
    name: str
    value: bool


@dataclass
class TimerNode(ASTNode):
    """Timer node."""
    name: str
    period: float
    settings: List[TimerSettingNode]


@dataclass
class RemapNode(ASTNode):
    """Remap node."""
    from_topic: str
    to_topic: str


@dataclass
class NamespaceNode(ASTNode):
    """Namespace node."""
    namespace: str


@dataclass
class FlagNode(ASTNode):
    """Flag node."""
    name: str
    value: bool


@dataclass
class QoSSettingNode(ASTNode):
    """QoS setting node."""
    name: str
    value: Union[str, int]


@dataclass
class QoSNode(ASTNode):
    """QoS configuration node."""
    settings: List[QoSSettingNode]

    @property
    def reliability(self):
        return self._get_setting('reliability')
    @property
    def durability(self):
        return self._get_setting('durability')
    @property
    def history(self):
        return self._get_setting('history')
    @property
    def depth(self):
        return self._get_setting('depth')
    @property
    def deadline(self):
        return self._get_setting('deadline')
    @property
    def lifespan(self):
        return self._get_setting('lifespan')
    @property
    def liveliness(self):
        return self._get_setting('liveliness')
    @property
    def lease_duration(self):
        return self._get_setting('lease_duration')

    def _get_setting(self, name):
        for s in self.settings:
            if s.name == name:
                return s.value
        return None


@dataclass
class PublisherNode(ASTNode):
    """Publisher node."""
    topic: str
    msg_type: str
    qos: Optional[QoSNode]


@dataclass
class SubscriberNode(ASTNode):
    """Subscriber node."""
    topic: str
    msg_type: str
    qos: Optional[QoSNode]


@dataclass
class ServiceNode(ASTNode):
    """Service node."""
    service: str
    srv_type: str
    qos: Optional[QoSNode]


@dataclass
class ClientNode(ASTNode):
    """Client node."""
    service: str
    srv_type: str
    qos: Optional[QoSNode]


@dataclass
class ActionNode(ASTNode):
    """Action node."""
    name: str
    action_type: str
    qos: Optional[QoSNode]


@dataclass
class CppMethodNode(ASTNode):
    """C++ method node."""
    name: str
    code: str


@dataclass
class NodeContentNode(ASTNode):
    """Node content node."""
    parameters: List[ParameterNode] = field(default_factory=list)
    lifecycle: Optional[LifecycleNode] = None
    timers: List[TimerNode] = field(default_factory=list)
    remaps: List[RemapNode] = field(default_factory=list)
    namespace: Optional[NamespaceNode] = None
    flags: List[FlagNode] = field(default_factory=list)
    publishers: List[PublisherNode] = field(default_factory=list)
    subscribers: List[SubscriberNode] = field(default_factory=list)
    services: List[ServiceNode] = field(default_factory=list)
    clients: List[ClientNode] = field(default_factory=list)
    actions: List[ActionNode] = field(default_factory=list)
    cpp_methods: List[CppMethodNode] = field(default_factory=list)


@dataclass
class NodeNode(ASTNode):
    """Node definition node."""
    name: str
    content: NodeContentNode


@dataclass
class KernelParamNode(ASTNode):
    """CUDA kernel parameter node."""
    direction: KernelParameterDirection
    param_type: str
    param_name: Optional[str]
    size_expr: Optional[str]


@dataclass
class KernelContentNode(ASTNode):
    """CUDA kernel content node."""
    block_size: Optional[tuple] = None
    grid_size: Optional[tuple] = None
    shared_memory: Optional[int] = None
    use_thrust: bool = False
    parameters: List[KernelParamNode] = field(default_factory=list)
    code: str = ""
    includes: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)


@dataclass
class KernelNode(ASTNode):
    """CUDA kernel node."""
    name: str
    content: KernelContentNode


@dataclass
class CudaKernelsNode(ASTNode):
    """CUDA kernels block node."""
    kernels: List[KernelNode]


@dataclass
class RoboDSLAST(ASTNode):
    """Root AST node."""
    includes: List[IncludeNode] = field(default_factory=list)
    nodes: List[NodeNode] = field(default_factory=list)
    cuda_kernels: Optional[CudaKernelsNode] = None 