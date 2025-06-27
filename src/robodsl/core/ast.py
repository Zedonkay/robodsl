"""Abstract Syntax Tree for RoboDSL.

This module defines the AST nodes for the RoboDSL language, providing
a clean representation of parsed configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Union, Set
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ast import KernelNode


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


class KernelParameterDirection(Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"


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
    type: str
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
    """Enhanced C++ method node with input/output parameters."""
    name: str
    inputs: List['MethodParamNode'] = field(default_factory=list)
    outputs: List['MethodParamNode'] = field(default_factory=list)
    code: str = ""


@dataclass
class MethodParamNode(ASTNode):
    """Method parameter node."""
    param_type: str
    param_name: str
    size_expr: Optional[str] = None
    is_const: bool = False


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
    cuda_kernels: List['KernelNode'] = field(default_factory=list)
    onnx_models: List['OnnxModelNode'] = field(default_factory=list)  # ONNX models within nodes


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
    is_const: bool = False


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
    cuda_kernels: Optional[CudaKernelsNode] = None  # Standalone kernels outside nodes 
    onnx_models: List['OnnxModelNode'] = field(default_factory=list)  # ONNX models
    pipelines: List['PipelineNode'] = field(default_factory=list)  # Pipeline definitions


# ONNX Model AST Nodes (Phase 3)
@dataclass
class InputDefNode(ASTNode):
    """ONNX model input definition node."""
    name: str
    type: str


@dataclass
class OutputDefNode(ASTNode):
    """ONNX model output definition node."""
    name: str
    type: str


@dataclass
class DeviceNode(ASTNode):
    """ONNX model device configuration node."""
    device: str  # "cpu" or "cuda"


@dataclass
class OptimizationNode(ASTNode):
    """ONNX model optimization configuration node."""
    optimization: str  # "tensorrt" or "openvino"


@dataclass
class ModelConfigNode(ASTNode):
    """ONNX model configuration node."""
    inputs: List[InputDefNode] = field(default_factory=list)
    outputs: List[OutputDefNode] = field(default_factory=list)
    device: Optional[DeviceNode] = None
    optimizations: List[OptimizationNode] = field(default_factory=list)


@dataclass
class OnnxModelNode(ASTNode):
    """ONNX model node."""
    name: str
    config: ModelConfigNode 


# Pipeline AST Nodes (Phase 4)
@dataclass
class StageInputNode(ASTNode):
    """Pipeline stage input node."""
    input_name: str


@dataclass
class StageOutputNode(ASTNode):
    """Pipeline stage output node."""
    output_name: str


@dataclass
class StageMethodNode(ASTNode):
    """Pipeline stage method node."""
    method_name: str


@dataclass
class StageModelNode(ASTNode):
    """Pipeline stage model node."""
    model_name: str


@dataclass
class StageTopicNode(ASTNode):
    """Pipeline stage topic node."""
    topic_path: str


@dataclass
class StageCudaKernelNode(ASTNode):
    """Pipeline stage CUDA kernel node."""
    kernel_name: str


@dataclass
class StageOnnxModelNode(ASTNode):
    """Pipeline stage ONNX model node."""
    model_name: str


@dataclass
class StageContentNode(ASTNode):
    """Pipeline stage content node."""
    inputs: List[StageInputNode] = field(default_factory=list)
    outputs: List[StageOutputNode] = field(default_factory=list)
    methods: List[StageMethodNode] = field(default_factory=list)
    models: List[StageModelNode] = field(default_factory=list)
    topics: List[StageTopicNode] = field(default_factory=list)
    cuda_kernels: List[StageCudaKernelNode] = field(default_factory=list)
    onnx_models: List[StageOnnxModelNode] = field(default_factory=list)


@dataclass
class StageNode(ASTNode):
    """Pipeline stage node."""
    name: str
    content: StageContentNode


@dataclass
class PipelineContentNode(ASTNode):
    """Pipeline content node."""
    stages: List[StageNode] = field(default_factory=list)


@dataclass
class PipelineNode(ASTNode):
    """Pipeline definition node."""
    name: str
    content: PipelineContentNode 