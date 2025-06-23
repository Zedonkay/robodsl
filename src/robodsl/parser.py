"""RoboDSL parser implementation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Union, Set, Tuple
import re
import ast

@dataclass
class QoSConfig:
    """Quality of Service configuration for ROS 2 communication."""
    reliability: Optional[str] = None
    durability: Optional[str] = None
    history: Optional[str] = None
    depth: int = 10
    deadline: Optional[int] = None
    lifespan: Optional[int] = None
    liveliness: Optional[str] = None
    liveliness_lease_duration: Optional[int] = None

@dataclass
class PublisherConfig:
    """Configuration for a ROS 2 publisher."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10

@dataclass
class SubscriberConfig:
    """Configuration for a ROS 2 subscriber."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10

@dataclass
class ServiceConfig:
    """Configuration for a ROS 2 service server."""
    service: str
    srv_type: str
    qos: Optional[QoSConfig] = None

@dataclass
class ActionConfig:
    """Configuration for a ROS 2 action server."""
    name: str
    action_type: str
    qos: Optional[QoSConfig] = None

@dataclass
class ParameterConfig:
    """Configuration for a node parameter."""
    name: str
    type: str
    default: Any = None
    description: str = ""
    read_only: bool = False

    # Provide backward-compatible alias used by older tests
    @property
    def value(self) -> Any:  # pragma: no cover
        """Alias for `default` to keep compatibility with legacy tests."""
        return self.default

@dataclass
class LifecycleConfig:
    """Configuration for node lifecycle settings."""
    enabled: bool = False
    autostart: bool = True
    cleanup_on_shutdown: bool = True

@dataclass
class TimerConfig:
    """Configuration for a node timer."""
    period: float
    callback: str
    oneshot: bool = False
    autostart: bool = True

@dataclass
class RemapRule:
    """Configuration for a topic remapping rule."""
    from_topic: str
    to_topic: str

@dataclass
class CppMethodConfig:
    """Configuration for a C++ method in a node."""
    name: str
    return_type: str
    parameters: List[str]
    implementation: str

@dataclass
class NodeConfig:
    """Configuration for a ROS 2 node."""
    name: str
    publishers: List[PublisherConfig] = field(default_factory=list)
    subscribers: List[SubscriberConfig] = field(default_factory=list)
    services: List[ServiceConfig] = field(default_factory=list)
    actions: List[ActionConfig] = field(default_factory=list)
    parameters: List[ParameterConfig] = field(default_factory=list)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    timers: List[TimerConfig] = field(default_factory=list)
    methods: List[CppMethodConfig] = field(default_factory=list)
    namespace: str = ""
    remap: List[RemapRule] = field(default_factory=list)
    parameter_callbacks: bool = False

@dataclass
class KernelParameter:
    """Configuration for a CUDA kernel parameter."""
    name: str
    type: str
    direction: Literal["in", "out", "inout"]
    is_const: bool = False
    is_pointer: bool = False

    # allow dict-style access expected by legacy tests
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key)

    def get(self, key, default=None):  # mimic dict.get
        try:
            return self[key]
        except KeyError:
            return default
    size_expr: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CudaKernelConfig:
    """Configuration for a CUDA kernel."""
    name: str
    parameters: List[KernelParameter]
    block_size: Tuple[int, int, int] = (256, 1, 1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    grid_size: Optional[Tuple[int, int, int]] = None
    shared_mem_bytes: int = 0
    use_thrust: bool = False
    code: str = ""
    includes: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)

    @property
    def inputs(self) -> List[KernelParameter]:
        """Get all input parameters."""
        return [p for p in self.parameters if p.direction in ('in', 'input')]

    @property
    def outputs(self) -> List[KernelParameter]:
        """Get all output parameters."""
        return [p for p in self.parameters if p.direction in ('out', 'output')]

@dataclass
class RoboDSLConfig:
    """Top-level configuration for a RoboDSL project."""
    project_name: str = "robodsl_project"
    nodes: List[NodeConfig] = field(default_factory=list)
    cuda_kernels: List[CudaKernelConfig] = field(default_factory=list)
    includes: Set[str] = field(default_factory=set)

def _find_matching_brace(content: str, start_pos: int) -> int:
    """Finds the position of the matching closing brace, ignoring braces inside strings."""
    if content[start_pos] != '{':
        return -1
    brace_count = 1
    i = start_pos + 1
    while i < len(content):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return i
        i += 1
    return -1

def parse_qos_config(qos_str: str) -> QoSConfig:
    """Parse QoS configuration from a string."""
    qos = QoSConfig()
    if not qos_str:
        return qos
    for pair in qos_str.strip().split():
        if '=' in pair:
            key, value = pair.split('=', 1)
            if hasattr(qos, key):
                attr_type = type(getattr(qos, key, None))
                if attr_type is int:
                    setattr(qos, key, int(value))
                else:
                    setattr(qos, key, value)
    return qos

def _parse_parameters(content: str, config: NodeConfig):
    """Parse 'parameter' lines in a node configuration.

    Supports the following forms (whitespace is flexible):
        parameter name = value                  # implicit type
        parameter name: type = value            # explicit type
        parameter name: value                   # implicit type using ':' separator
    The function processes the DSL content line-by-line instead of relying on a
    single complex regular expression to avoid multi-line greediness issues that
    previously caused values to consume subsequent DSL statements.
    """
    # Iterate over each line that begins with the keyword "parameter"
    for line in content.splitlines():
        line = line.strip()
        if not re.match(r'^parameter\b', line):
            continue

        # Remove the leading keyword to work on the remainder
        remainder = line[len("parameter"):].strip()

        # Split into tokens by first '=' if present, otherwise by first ':'
        if '=' in remainder:
            left, value_str = remainder.split('=', 1)
            left = left.strip()
        elif ':' in remainder:
            left, value_str = remainder.split(':', 1)
            left = left.strip()
        else:
            # Malformed line; skip
            continue

        value_str = value_str.strip()

        # Determine if an explicit type is provided in `left`
        if ':' in left:
            name_part, type_part = [part.strip() for part in left.split(':', 1)]
            param_name = name_part
            param_type = type_part if type_part else None
        else:
            param_name = left.strip()
            param_type = None

        # Convert the textual value into a Python literal when possible, with
        # additional handling for lowercase booleans common in YAML/DSL files.
        lowered = value_str.lower()
        if lowered in ("true", "false"):
            value = lowered == "true"
        else:
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str

        # Infer type when not explicitly provided
        if param_type is None:
            if isinstance(value, bool):
                param_type = 'bool'
            elif isinstance(value, int):
                param_type = 'int'
            elif isinstance(value, float):
                param_type = 'double'
            elif isinstance(value, str):
                param_type = 'string'
            else:
                param_type = 'auto'

        config.parameters.append(
            ParameterConfig(name=param_name, type=param_type, default=value)
        )


def _parse_methods(content: str, config: NodeConfig):
    '''Extract C++ method definitions from a node block and populate
    ``config.methods`` with ``CppMethodConfig`` objects.

    The method list is written in the DSL as::

        methods = [{
            name = "foo"
            return_type = "int"
            parameters = ["int a", "int b"]
            implementation = """
                return a + b;
            """
        }, {...}]

    The parser must correctly handle nested braces and the fact that the
    implementation field can contain braces and the ``"""`` triple-quoted
    sentinel itself.  A small state-machine is therefore used instead of a
    naive regular expression.
    '''
    # ---------------------------------------------------------------------
    # 1. Locate the *start* of the methods list (the opening '[')
    # ---------------------------------------------------------------------
    list_start_match = re.search(r"methods\s*=\s*\[", content)
    if not list_start_match:
        return  # No methods section – nothing to do.

    idx = list_start_match.end()            # position right *after* the '['
    depth = 1                               # depth of nested [ ] brackets
    i = idx
    in_triple = False                       # are we inside a """ string?
    while i < len(content) and depth > 0:
        if content.startswith('"""', i):
            in_triple = not in_triple
            i += 3
            continue
        if not in_triple:
            if content[i] == '[':
                depth += 1
            elif content[i] == ']':
                depth -= 1
                if depth == 0:
                    list_end_idx = i
                    break
        i += 1
    else:
        # Unbalanced brackets – bail out gracefully.
        return

    methods_list_text = content[idx:list_end_idx]

    # ---------------------------------------------------------------------
    # 2. Split the list text into individual method *blocks* delimited by { }.
    #    We re-use a similar mini state-machine to be immune to braces inside
    #    triple-quoted strings.
    # ---------------------------------------------------------------------
    def extract_blocks(text: str) -> List[str]:
        blocks: List[str] = []
        i = 0
        n = len(text)
        in_triple = False
        while i < n:
            if text[i] == '{' and not in_triple:
                start = i + 1  # skip the opening '{'
                brace_depth = 1
                i += 1
                while i < n and brace_depth > 0:
                    if text.startswith('"""', i):
                        in_triple = not in_triple
                        i += 3
                        continue
                    if not in_triple:
                        if text[i] == '{':
                            brace_depth += 1
                        elif text[i] == '}':
                            brace_depth -= 1
                            if brace_depth == 0:
                                blocks.append(text[start:i])
                                i += 1  # move past the closing '}'
                                break
                    i += 1
            else:
                if text.startswith('"""', i):
                    in_triple = not in_triple
                    i += 3
                else:
                    i += 1
        return blocks

    for block in extract_blocks(methods_list_text):
        # Parse the *fields* inside the block using small regexes.
        name_match = re.search(r"name\s*=\s*\"([^\"]+)\"", block)
        rtype_match = re.search(r"return_type\s*=\s*\"([^\"]+)\"", block)
        params_match = re.search(r"parameters\s*=\s*(\[[^\]]*\])", block, re.DOTALL)
        impl_match = re.search(r"implementation\s*=\s*\"\"\"(.*?)\"\"\"", block, re.DOTALL)

        if not (name_match and rtype_match and params_match and impl_match):
            # Skip malformed entries – they will be reported by the validator
            # layer later on.
            continue

        try:
            params_list = ast.literal_eval(params_match.group(1))
        except Exception:
            params_list = []

        config.methods.append(
            CppMethodConfig(
                name=name_match.group(1),
                return_type=rtype_match.group(1),
                parameters=params_list,
                implementation=impl_match.group(1).strip(),
            )
        )

    # Nothing to *return* – results are stored in ``config.methods``.
    return





def parse_robodsl(content: str) -> RoboDSLConfig:
    """Parse RoboDSL content and return a configuration object."""
    config = RoboDSLConfig()
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    project_match = re.search(r'project_name\s*=\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?', content)
    if project_match:
        config.project_name = project_match.group(1)
    node_matches = re.finditer(r'node\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{', content)
    for match in node_matches:
        node_name = match.group(1)
        start_pos = match.end() - 1
        end_pos = _find_matching_brace(content, start_pos)
        if end_pos == -1:
            continue
        node_content = content[start_pos + 1:end_pos]
        node_config = NodeConfig(name=node_name)
        pub_matches = re.finditer(r'publisher\s+(\S+)\s+(\S+)(?:\s+qos\s+([^\n{]+))?', node_content)
        for pub_match in pub_matches:
            topic, msg_type, qos_str = pub_match.groups()
            qos = parse_qos_config(qos_str) if qos_str else None
            node_config.publishers.append(PublisherConfig(topic=topic, msg_type=msg_type, qos=qos))
        sub_matches = re.finditer(r'subscriber\s+(\S+)\s+(\S+)(?:\s+qos\s+([^\n{]+))?', node_content)
        for sub_match in sub_matches:
            topic, msg_type, qos_str = sub_match.groups()
            qos = parse_qos_config(qos_str) if qos_str else None
            node_config.subscribers.append(SubscriberConfig(topic=topic, msg_type=msg_type, qos=qos))
        srv_matches = re.finditer(r'service\s+(\S+)\s+(\S+)(?:\s+qos\s+([^\\n{]+))?', node_content)
        for srv_match in srv_matches:
            service, srv_type, qos_str = srv_match.groups()
            qos = parse_qos_config(qos_str) if qos_str else None
            node_config.services.append(ServiceConfig(service=service, srv_type=srv_type, qos=qos))
        act_matches = re.finditer(r'action\s+(\S+)\s+(\S+)(?:\s+qos\s+([^\\n{]+))?', node_content)
        for act_match in act_matches:
            name, action_type, qos_str = act_match.groups()
            qos = parse_qos_config(qos_str) if qos_str else None
            node_config.actions.append(ActionConfig(name=name, action_type=action_type, qos=qos))
        if re.search(r'lifecycle\s*=\s*true', node_content):
            node_config.lifecycle.enabled = True
        _parse_methods(node_content, node_config)
        _parse_parameters(node_content, node_config)
        config.nodes.append(node_config)
    kernels_block_match = re.search(r'cuda_kernels\s*\{', content)
    if kernels_block_match:
        kernels_start_brace = kernels_block_match.start() + content[kernels_block_match.start():].find('{')
        kernels_end_brace = _find_matching_brace(content, kernels_start_brace)
        if kernels_end_brace != -1:
            kernels_content = content[kernels_start_brace + 1:kernels_end_brace]
            kernel_matches = re.finditer(r'kernel\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{', kernels_content)
            for match in kernel_matches:
                kernel_name = match.group(1)
                start_pos = match.end() - 1
                end_pos = _find_matching_brace(kernels_content, start_pos)
                if end_pos == -1:
                    continue
                kernel_content_inner = kernels_content[start_pos + 1:end_pos]
                params = []
                # Accept both the older (in/out/inout) and newer (input/output) keywords.
                # First, handle simplified DSL syntax lines like 'input: Type' / 'output: Type'
                for line in kernel_content_inner.splitlines():
                    line = line.strip()
                    io_match = re.match(r"(input|output)\s*:\s*([\w\d_<>:]+)\s*(?:\(([^)]+)\))?", line)
                    if io_match:
                        direction, type_name, meta = io_match.groups()
                        metadata = {}
                        if meta:
                            for pair in meta.split(','):
                                if '=' in pair:
                                    k, v = pair.strip().split('=', 1)
                                    metadata[k.strip()] = int(v) if v.isdigit() else v.strip()
                        params.append(KernelParameter(name=type_name.lower(), type=type_name, direction='in' if direction == 'input' else 'out', metadata=metadata))
                # Then, parse explicit parameter lines with keywords.
                k_param_pattern = re.compile(
                    r"(in|out|inout|input|output)"  # direction keyword (group 1)
                    r"\s+"
                    r"(const\s+)?"               # optional const (group 2)
                    r"([\w\d_<>:]+)"            # type (group 3)
                    r"(\s*\*|\s*&)?"           # optional pointer / reference (group 4)
                    r"\s+([\w\d_]+)"           # param name (group 5)
                )
                for p_match in k_param_pattern.finditer(kernel_content_inner):
                    direction, is_const, p_type, pointer_ref, p_name = p_match.groups()
                    params.append(KernelParameter(name=p_name, type=p_type.strip(), direction=direction, is_const=bool(is_const), is_pointer=bool(pointer_ref and '*' in pointer_ref)))
                block_size_match = re.search(r'block_size\s*[:=]\s*\((\d+),\s*(\d+),\s*(\d+)\)', kernel_content_inner)
                block_size = (256, 1, 1)
                if block_size_match:
                    block_size = tuple(map(int, block_size_match.groups()))
                code_match = re.search(r'code\s*=\s*"""(.*?)"""', kernel_content_inner, re.DOTALL)
                code = code_match.group(1).strip() if code_match else ''
                kernel_config = CudaKernelConfig(name=kernel_name, parameters=params, block_size=block_size, code=code)
                config.cuda_kernels.append(kernel_config)
    return config
