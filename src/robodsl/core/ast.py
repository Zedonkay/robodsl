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
class ServicePrimitiveNode(ASTNode):
    """Service primitive node."""
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
    default_value: Optional[str] = None


@dataclass
class NodeContentNode(ASTNode):
    """Node content node.
    - used_kernels: List of names of referenced global CUDA kernels (via use_kernel)
    """
    parameters: List[ParameterNode] = field(default_factory=list)
    lifecycle: Optional[LifecycleNode] = None
    timers: List[TimerNode] = field(default_factory=list)
    remaps: List[RemapNode] = field(default_factory=list)
    namespace: Optional[NamespaceNode] = None
    flags: List[FlagNode] = field(default_factory=list)
    publishers: List[PublisherNode] = field(default_factory=list)
    subscribers: List[SubscriberNode] = field(default_factory=list)
    services: List[ServicePrimitiveNode] = field(default_factory=list)
    clients: List[ClientNode] = field(default_factory=list)
    actions: List[ActionNode] = field(default_factory=list)
    cpp_methods: List[CppMethodNode] = field(default_factory=list)
    cuda_kernels: List['KernelNode'] = field(default_factory=list)
    onnx_models: List['OnnxModelNode'] = field(default_factory=list)  # ONNX models within nodes
    used_kernels: List[str] = field(default_factory=list)  # Referenced global CUDA kernels
    raw_cpp_code: List['RawCppCodeNode'] = field(default_factory=list)  # Raw C++ code within nodes


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
    kernel_parameters: List[dict] = field(default_factory=list)  # Kernel parameter definitions
    code: str = ""
    cuda_includes: List[str] = field(default_factory=list)
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


# Data Structure AST Nodes
@dataclass
class StructMemberNode(ASTNode):
    """Struct member node."""
    type: str
    name: str
    array_spec: Optional[str] = None


@dataclass
class StructContentNode(ASTNode):
    """Struct content node."""
    members: List[StructMemberNode] = field(default_factory=list)
    methods: List[CppMethodNode] = field(default_factory=list)
    includes: List[IncludeNode] = field(default_factory=list)


@dataclass
class StructNode(ASTNode):
    """Struct definition node."""
    name: str
    content: StructContentNode


@dataclass
class InheritanceNode(ASTNode):
    """Inheritance specification node."""
    base_classes: List[tuple[str, str]]  # List of (access_specifier, class_name) tuples


@dataclass
class AccessSectionNode(ASTNode):
    """Access section node (public, private, protected)."""
    access_specifier: str  # "public", "private", "protected"
    members: List[StructMemberNode] = field(default_factory=list)
    methods: List[CppMethodNode] = field(default_factory=list)


@dataclass
class ClassContentNode(ASTNode):
    """Class content node."""
    access_sections: List[AccessSectionNode] = field(default_factory=list)
    members: List[StructMemberNode] = field(default_factory=list)
    methods: List[CppMethodNode] = field(default_factory=list)
    includes: List[IncludeNode] = field(default_factory=list)


@dataclass
class ClassNode(ASTNode):
    """Class definition node."""
    name: str
    content: ClassContentNode
    inheritance: Optional[InheritanceNode] = None


@dataclass
class EnumValueNode(ASTNode):
    """Enum value node."""
    name: str
    value: Optional[str] = None


@dataclass
class EnumContentNode(ASTNode):
    """Enum content node."""
    values: List[EnumValueNode] = field(default_factory=list)


@dataclass
class EnumNode(ASTNode):
    """Enum definition node."""
    name: str
    content: EnumContentNode
    enum_type: Optional[str] = None  # "class" or "struct" for enum class


@dataclass
class TypedefNode(ASTNode):
    """Typedef definition node."""
    original_type: str
    new_name: str


@dataclass
class UsingNode(ASTNode):
    """Using declaration node."""
    original_type: str
    new_name: str


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


# Pythonic Class AST Nodes



# Custom Message/Service/Action AST Nodes
@dataclass
class MessageFieldNode(ASTNode):
    """Message field node."""
    type: str
    name: str
    array_spec: Optional[str] = None
    default_value: Optional[ValueNode] = None


@dataclass
class MessageContentNode(ASTNode):
    """Message content node."""
    fields: List[MessageFieldNode] = field(default_factory=list)
    constants: List[MessageFieldNode] = field(default_factory=list)  # For constants like "uint8 FOO=1"


@dataclass
class MessageNode(ASTNode):
    """Custom message definition node."""
    name: str
    content: MessageContentNode


@dataclass
class ServiceRequestNode(ASTNode):
    """Service request node."""
    fields: List[MessageFieldNode] = field(default_factory=list)


@dataclass
class ServiceResponseNode(ASTNode):
    """Service response node."""
    fields: List[MessageFieldNode] = field(default_factory=list)


@dataclass
class ServiceContentNode(ASTNode):
    """Service content node."""
    request: ServiceRequestNode
    response: ServiceResponseNode


@dataclass
class ServiceNode(ASTNode):
    """Custom service definition node."""
    name: str
    content: ServiceContentNode


@dataclass
class ActionGoalNode(ASTNode):
    """Action goal node."""
    fields: List[MessageFieldNode] = field(default_factory=list)


@dataclass
class ActionFeedbackNode(ASTNode):
    """Action feedback node."""
    fields: List[MessageFieldNode] = field(default_factory=list)


@dataclass
class ActionResultNode(ASTNode):
    """Action result node."""
    fields: List[MessageFieldNode] = field(default_factory=list)


@dataclass
class ActionContentNode(ASTNode):
    """Action content node."""
    goal: ActionGoalNode
    feedback: ActionFeedbackNode
    result: ActionResultNode


@dataclass
class CustomActionNode(ASTNode):
    """Custom action definition node."""
    name: str
    content: ActionContentNode


# Dynamic Runtime AST Nodes
@dataclass
class DynamicParameterNode(ASTNode):
    """Dynamic parameter node."""
    name: str
    type: str
    default_value: ValueNode
    min_value: Optional[ValueNode] = None
    max_value: Optional[ValueNode] = None
    step: Optional[ValueNode] = None
    description: Optional[str] = None


@dataclass
class DynamicRemapNode(ASTNode):
    """Dynamic remap node."""
    from_topic: str
    to_topic: str
    condition: Optional[str] = None  # Optional condition for when to apply


# Simulation Configuration AST Nodes
@dataclass
class SimulationPluginNode(ASTNode):
    """Simulation plugin node."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationWorldNode(ASTNode):
    """Simulation world configuration node."""
    world_file: str
    physics_engine: str = "ode"  # ode, bullet, etc.
    gravity: tuple = (0, 0, -9.81)
    max_step_size: float = 0.001
    real_time_factor: float = 1.0


@dataclass
class SimulationRobotNode(ASTNode):
    """Simulation robot configuration node."""
    model_file: str  # URDF/SDF file
    namespace: Optional[str] = None
    initial_pose: Optional[tuple] = None  # (x, y, z, roll, pitch, yaw)
    plugins: List[SimulationPluginNode] = field(default_factory=list)


@dataclass
class SimulationConfigNode(ASTNode):
    """Simulation configuration node."""
    simulator: str  # "gazebo", "isaac_sim", etc.
    world: Optional[SimulationWorldNode] = None
    robots: List[SimulationRobotNode] = field(default_factory=list)
    plugins: List[SimulationPluginNode] = field(default_factory=list)
    gui: bool = True
    headless: bool = False
    physics_engine: str = "ode"


@dataclass
class HardwareInLoopNode(ASTNode):
    """Hardware-in-the-loop configuration node."""
    simulation_nodes: List[str] = field(default_factory=list)  # Node names that run in sim
    hardware_nodes: List[str] = field(default_factory=list)    # Node names that run on hardware
    bridge_config: Optional[str] = None  # Optional bridge configuration file


@dataclass
class RawCppCodeNode(ASTNode):
    """Raw C++ code node that gets passed through as-is."""
    code: str
    location: str = "global"  # "global" for outside nodes, "node" for inside nodes


# Advanced C++ Features AST Nodes (Phase 8)

@dataclass
class TemplateParamNode(ASTNode):
    """Template parameter node."""
    name: str
    param_type: str  # "typename" or "class"
    default_value: Optional[str] = None


@dataclass
class TemplateStructNode(ASTNode):
    """Template struct definition node."""
    name: str
    template_params: List[TemplateParamNode]
    content: StructContentNode


@dataclass
class TemplateClassNode(ASTNode):
    """Template class definition node."""
    name: str
    template_params: List[TemplateParamNode]
    content: ClassContentNode
    inheritance: Optional[InheritanceNode] = None


@dataclass
class TemplateFunctionNode(ASTNode):
    """Template function definition node."""
    name: str
    template_params: List[TemplateParamNode]
    parameters: List[MethodParamNode]
    return_type: Optional[str] = None
    code: str = ""


@dataclass
class TemplateAliasNode(ASTNode):
    """Template alias definition node."""
    name: str
    template_params: List[TemplateParamNode]
    aliased_type: str


@dataclass
class StaticAssertNode(ASTNode):
    """Static assertion node."""
    condition: str
    message: str


@dataclass
class GlobalConstexprNode(ASTNode):
    """Global constexpr variable node."""
    name: str
    type: str
    value: ValueNode


@dataclass
class GlobalDeviceConstNode(ASTNode):
    """Global device constant node."""
    name: str
    type: str
    array_size: str
    values: List[ValueNode]


@dataclass
class GlobalStaticInlineNode(ASTNode):
    """Global static inline function node."""
    name: str
    parameters: List[MethodParamNode]
    return_type: Optional[str] = None
    code: str = ""


@dataclass
class OperatorOverloadNode(ASTNode):
    """Operator overload node."""
    operator: str
    parameters: List[MethodParamNode]
    return_type: Optional[str] = None
    code: str = ""


@dataclass
class ConstructorNode(ASTNode):
    """Constructor definition node."""
    parameters: List[MethodParamNode]
    member_initializers: List[tuple[str, str]] = field(default_factory=list)  # (member, value)
    code: str = ""


@dataclass
class DestructorNode(ASTNode):
    """Destructor definition node."""
    code: str = ""


@dataclass
class BitfieldMemberNode(ASTNode):
    """Bitfield member node."""
    type: str
    name: str
    bits: int


@dataclass
class BitfieldNode(ASTNode):
    """Bitfield definition node."""
    name: str
    members: List[BitfieldMemberNode]


@dataclass
class PreprocessorDirectiveNode(ASTNode):
    """Preprocessor directive node."""
    directive_type: str  # "pragma", "include", "if", "define", "error", "line"
    content: str


@dataclass
class FunctionAttributeNode(ASTNode):
    """Function with attributes node."""
    attributes: List[str]  # List of attribute names
    name: str
    parameters: List[MethodParamNode]
    return_type: Optional[str] = None
    code: str = ""


@dataclass
class ConceptRequiresNode(ASTNode):
    """Concept requires clause node."""
    type_param: str
    requirements: List[str]  # List of required operations


@dataclass
class ConceptNode(ASTNode):
    """Concept definition node."""
    name: str
    requires: ConceptRequiresNode


@dataclass
class FriendDeclarationNode(ASTNode):
    """Friend declaration node."""
    friend_type: str  # "class" or "function"
    target: str  # Class name or function declaration


@dataclass
class UserDefinedLiteralNode(ASTNode):
    """User-defined literal node."""
    literal_suffix: str
    return_type: str
    code: str = ""


@dataclass
class RoboDSLAST(ASTNode):
    """Root AST node."""
    includes: List[IncludeNode] = field(default_factory=list)
    data_structures: List[Union[StructNode, ClassNode, EnumNode, TypedefNode, UsingNode]] = field(default_factory=list)
    nodes: List[NodeNode] = field(default_factory=list)
    cuda_kernels: Optional[CudaKernelsNode] = None  # Standalone kernels outside nodes 
    onnx_models: List['OnnxModelNode'] = field(default_factory=list)  # ONNX models
    pipelines: List['PipelineNode'] = field(default_factory=list)  # Pipeline definitions
    # Custom message/service/action types
    messages: List[MessageNode] = field(default_factory=list)
    services: List[ServiceNode] = field(default_factory=list)
    actions: List[CustomActionNode] = field(default_factory=list)
    # Dynamic runtime configuration
    dynamic_parameters: List[DynamicParameterNode] = field(default_factory=list)
    dynamic_remaps: List[DynamicRemapNode] = field(default_factory=list)
    # Simulation configuration
    simulation: Optional[SimulationConfigNode] = None
    hil_config: Optional[HardwareInLoopNode] = None
    # Raw C++ code blocks
    raw_cpp_code: List[RawCppCodeNode] = field(default_factory=list)  # Global C++ code outside nodes
    # Advanced C++ features
    advanced_cpp_features: List[Union[TemplateStructNode, TemplateClassNode, TemplateFunctionNode, TemplateAliasNode, 
                                     StaticAssertNode, GlobalConstexprNode, GlobalDeviceConstNode, GlobalStaticInlineNode,
                                     OperatorOverloadNode, ConstructorNode, DestructorNode, BitfieldNode, 
                                     PreprocessorDirectiveNode, FunctionAttributeNode, ConceptNode, 
                                     FriendDeclarationNode, UserDefinedLiteralNode]] = field(default_factory=list) 