import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    DidChangeConfigurationNotification,
    CompletionItem,
    CompletionItemKind,
    TextDocumentPositionParams,
    TextDocumentSyncKind,
    InitializeResult,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Position,
    Hover,
    HoverParams,
    Definition,
    DefinitionParams,
    Location,
    SymbolInformation,
    SymbolKind,
    DocumentSymbol,
    DocumentSymbolParams
} from 'vscode-languageserver/node';

import {
    TextDocument
} from 'vscode-languageserver-textdocument';

// Create a connection for the server
const connection = createConnection(ProposedFeatures.all);

// Create a text document manager
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;

// Keywords and completions
const keywords = [
    'node', 'cuda_kernels', 'kernel', 'method', 'parameter', 'remap', 'namespace', 
    'flag', 'lifecycle', 'timer', 'client', 'publisher', 'subscriber', 'service', 
    'action', 'include', 'input', 'output', 'code', 'in', 'out', 'inout', 
    'block_size', 'grid_size', 'shared_memory', 'use_thrust', 'qos', 'onnx_model', 
    'device', 'optimization', 'pipeline', 'stage', 'config', 'true', 'false'
];

const rosTypes = [
    'std_msgs/String', 'std_msgs/Int32', 'std_msgs/Float64', 'std_msgs/Bool',
    'geometry_msgs/Twist', 'geometry_msgs/Pose', 'geometry_msgs/Point',
    'sensor_msgs/Image', 'sensor_msgs/LaserScan', 'sensor_msgs/PointCloud2',
    'nav_msgs/Odometry', 'nav_msgs/Path', 'nav_msgs/OccupancyGrid'
];

const cppTypes = [
    'int', 'float', 'double', 'bool', 'char', 'string', 'std::string',
    'std::vector', 'std::array', 'std::map', 'std::unordered_map',
    'cv::Mat', 'cv::Point', 'cv::Point2f', 'cv::Point3f',
    'Eigen::Vector3d', 'Eigen::Matrix3d', 'Eigen::Quaterniond'
];

const qosSettings = [
    'reliability', 'durability', 'history', 'depth', 'deadline', 
    'lifespan', 'liveliness', 'liveliness_lease_duration'
];

const qosValues = [
    'reliable', 'best_effort', 'transient_local', 'volatile', 
    'keep_last', 'keep_all', 'automatic', 'manual_by_topic'
];

// Simple parser for basic syntax validation
class RoboDSLParser {
    private errors: Array<{line: number, column: number, message: string}> = [];
    
    parse(text: string): boolean {
        this.errors = [];
        const lines = text.split('\n');
        
        let braceCount = 0;
        let inNode = false;
        let inMethod = false;
        let inKernel = false;
        let inCudaKernels = false;
        let inOnnxModel = false;
        let inPipeline = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            const lineNumber = i + 1;
            
            // Skip comments and empty lines
            if (line.startsWith('//') || line === '' || line.startsWith('/*')) {
                continue;
            }
            
            // Count braces
            const openBraces = (line.match(/\{/g) || []).length;
            const closeBraces = (line.match(/\}/g) || []).length;
            braceCount += openBraces - closeBraces;
            
            // Check for syntax errors
            this.checkLineSyntax(line, lineNumber, i);
            
            // Track context
            if (line.includes('node ')) {
                inNode = true;
                inMethod = false;
                inKernel = false;
                inCudaKernels = false;
                inOnnxModel = false;
                inPipeline = false;
            } else if (line.includes('method ')) {
                inMethod = true;
                inKernel = false;
            } else if (line.includes('kernel ')) {
                inKernel = true;
                inMethod = false;
            } else if (line.includes('cuda_kernels')) {
                inCudaKernels = true;
                inNode = false;
                inMethod = false;
                inKernel = false;
                inOnnxModel = false;
                inPipeline = false;
            } else if (line.includes('onnx_model')) {
                inOnnxModel = true;
                inNode = false;
                inMethod = false;
                inKernel = false;
                inCudaKernels = false;
                inPipeline = false;
            } else if (line.includes('pipeline ')) {
                inPipeline = true;
                inNode = false;
                inMethod = false;
                inKernel = false;
                inCudaKernels = false;
                inOnnxModel = false;
            }
        }
        
        // Check for unmatched braces
        if (braceCount !== 0) {
            this.errors.push({
                line: lines.length,
                column: 0,
                message: `Unmatched braces: ${braceCount > 0 ? 'missing' : 'extra'} ${Math.abs(braceCount)} closing brace(s)`
            });
        }
        
        return this.errors.length === 0;
    }
    
    private checkLineSyntax(line: string, lineNumber: number, lineIndex: number): void {
        // Check for missing colons after keywords
        const keywordPattern = /\b(node|parameter|timer|remap|namespace|flag|publisher|subscriber|service|client|action|method|kernel|input|output|code|block_size|grid_size|shared_memory|use_thrust|device|optimization)\b/;
        const match = line.match(keywordPattern);
        
        if (match && !line.includes(':') && !line.includes('{') && !line.includes('=')) {
            const keyword = match[0];
            const column = line.indexOf(keyword);
            this.errors.push({
                line: lineNumber,
                column: column,
                message: `Missing colon (:) after keyword '${keyword}'`
            });
        }
        
        // Check for invalid syntax patterns
        if (line.includes('node') && !line.includes('{') && !line.includes(':')) {
            this.errors.push({
                line: lineNumber,
                column: line.indexOf('node'),
                message: 'Node definition must be followed by opening brace {'
            });
        }
        
        // Check for missing quotes in topic paths
        if (line.includes('publisher') || line.includes('subscriber') || line.includes('service') || line.includes('client') || line.includes('action')) {
            if (line.includes(':') && !line.includes('"')) {
                this.errors.push({
                    line: lineNumber,
                    column: line.indexOf(':'),
                    message: 'Topic type must be quoted'
                });
            }
        }
    }
    
    getErrors(): Array<{line: number, column: number, message: string}> {
        return this.errors;
    }
}

const parser = new RoboDSLParser();

connection.onInitialize((params: InitializeParams): InitializeResult => {
    const capabilities = params.capabilities;

    // Does the client support the `workspace/configuration` request?
    // If not, we will fall back using global settings.
    hasConfigurationCapability = !!(
        capabilities.workspace && !!capabilities.workspace.configuration
    );
    hasWorkspaceFolderCapability = !!(
        capabilities.workspace && !!capabilities.workspace.workspaceFolders
    );
    hasDiagnosticRelatedInformationCapability = !!(
        capabilities.textDocument &&
        capabilities.textDocument.publishDiagnostics &&
        capabilities.textDocument.publishDiagnostics.relatedInformation
    );

    const result: InitializeResult = {
        capabilities: {
            textDocumentSync: TextDocumentSyncKind.Incremental,
            // Tell the client that this server supports code completion.
            completionProvider: {
                resolveProvider: true,
                triggerCharacters: ['.', ':', ' ', '\n']
            },
            // Tell the client that this server supports hover.
            hoverProvider: true,
            // Tell the client that this server supports go to definition.
            definitionProvider: true,
            // Tell the client that this server supports document symbols.
            documentSymbolProvider: true
        }
    };
    if (hasWorkspaceFolderCapability) {
        result.capabilities.workspace = {
            workspaceFolders: {
                supported: true
            }
        };
    }
    return result;
});

connection.onInitialized(() => {
    if (hasConfigurationCapability) {
        // Register for all configuration changes.
        connection.client.register(DidChangeConfigurationNotification.type, undefined);
    }
    if (hasWorkspaceFolderCapability) {
        connection.workspace.onDidChangeWorkspaceFolders(_event => {
            connection.console.log('Workspace folder change event received.');
        });
    }
});

// The example settings
interface ExampleSettings {
    maxNumberOfProblems: number;
}

// The global settings, used when the `workspace/configuration` request is not supported by the client.
// Please note that this is not the case when using this server with the client provided in this example
// but could happen with other clients.
const defaultSettings: ExampleSettings = { maxNumberOfProblems: 1000 };
let globalSettings: ExampleSettings = defaultSettings;

// Cache the settings of all open documents
const documentSettings: Map<string, Promise<ExampleSettings>> = new Map();

connection.onDidChangeConfiguration(change => {
    if (hasConfigurationCapability) {
        // Reset all cached document settings
        documentSettings.clear();
    } else {
        globalSettings = <ExampleSettings>(
            (change.settings.robodslLanguageServer || defaultSettings)
        );
    }

    // Revalidate all open text documents
    documents.all().forEach(validateTextDocument);
});

function getDocumentSettings(resource: string): Promise<ExampleSettings> {
    if (!hasConfigurationCapability) {
        return Promise.resolve(globalSettings);
    }
    let result = documentSettings.get(resource);
    if (!result) {
        result = connection.workspace.getConfiguration({
            scopeUri: resource,
            section: 'robodslLanguageServer'
        });
        documentSettings.set(resource, result);
    }
    return result;
}

// Only keep settings for open documents
documents.onDidClose(e => {
    documentSettings.delete(e.document.uri);
});

// The content of a text document has changed. This event is emitted
// when the text document first opened or when its content has changed.
documents.onDidChangeContent(change => {
    validateTextDocument(change.document);
});

async function validateTextDocument(textDocument: TextDocument): Promise<void> {
    const settings = await getDocumentSettings(textDocument.uri);
    const text = textDocument.getText();
    const diagnostics: Diagnostic[] = [];

    // Parse the document
    const isValid = parser.parse(text);
    const errors = parser.getErrors();
    
    // Convert parser errors to diagnostics
    errors.forEach(error => {
        diagnostics.push({
            severity: DiagnosticSeverity.Error,
            range: {
                start: { line: error.line - 1, character: error.column },
                end: { line: error.line - 1, character: error.column + 1 }
            },
            message: error.message,
            source: 'robodsl'
        });
    });

    // Send the computed diagnostics to VSCode.
    connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });
}

connection.onDidChangeWatchedFiles(_change => {
    // Monitored files have change in VSCode
    connection.console.log('We received a file change event');
});

// This handler provides the initial list of completion items.
connection.onCompletion(
    (_textDocumentPosition: TextDocumentPositionParams): CompletionItem[] => {
        const completions: CompletionItem[] = [];
        
        // Add keywords
        keywords.forEach(keyword => {
            completions.push({
                label: keyword,
                kind: CompletionItemKind.Keyword,
                detail: `RoboDSL keyword: ${keyword}`,
                documentation: getKeywordDocumentation(keyword)
            });
        });
        
        // Add ROS types
        rosTypes.forEach(type => {
            completions.push({
                label: type,
                kind: CompletionItemKind.Class,
                detail: `ROS message type: ${type}`,
                documentation: `ROS message type for ${type.split('/')[1]}`
            });
        });
        
        // Add C++ types
        cppTypes.forEach(type => {
            completions.push({
                label: type,
                kind: CompletionItemKind.TypeParameter,
                detail: `C++ type: ${type}`,
                documentation: `C++ data type`
            });
        });
        
        // Add QoS settings
        qosSettings.forEach(setting => {
            completions.push({
                label: setting,
                kind: CompletionItemKind.Property,
                detail: `QoS setting: ${setting}`,
                documentation: `Quality of Service setting for ROS topics`
            });
        });
        
        // Add QoS values
        qosValues.forEach(value => {
            completions.push({
                label: value,
                kind: CompletionItemKind.Value,
                detail: `QoS value: ${value}`,
                documentation: `Quality of Service value`
            });
        });
        
        return completions;
    }
);

// This handler resolves additional information for the item selected in
// the completion list.
connection.onCompletionResolve(
    (item: CompletionItem): CompletionItem => {
        if (item.data === 1) {
            item.detail = 'TypeScript details';
            item.documentation = 'TypeScript documentation';
        } else if (item.data === 2) {
            item.detail = 'JavaScript details';
            item.documentation = 'JavaScript documentation';
        }
        return item;
    }
);

connection.onHover(
    (params: HoverParams): Hover | null => {
        const document = documents.get(params.textDocument.uri);
        if (!document) {
            return null;
        }

        const position = params.position;
        const text = document.getText();
        const lines = text.split('\n');
        const line = lines[position.line];
        
        // Find the word at the cursor position
        const wordRange = getWordRangeAtPosition(line, position.character);
        if (!wordRange) {
            return null;
        }
        
        const word = line.substring(wordRange.start, wordRange.end);
        
        // Provide hover information based on the word
        const hoverInfo = getHoverInformation(word);
        if (hoverInfo) {
            return {
                contents: {
                    kind: 'markdown',
                    value: hoverInfo
                },
                range: {
                    start: { line: position.line, character: wordRange.start },
                    end: { line: position.line, character: wordRange.end }
                }
            };
        }
        
        return null;
    }
);

connection.onDefinition(
    (params: DefinitionParams): Definition | null => {
        const document = documents.get(params.textDocument.uri);
        if (!document) {
            return null;
        }

        const position = params.position;
        const text = document.getText();
        const lines = text.split('\n');
        const line = lines[position.line];
        
        // Find the word at the cursor position
        const wordRange = getWordRangeAtPosition(line, position.character);
        if (!wordRange) {
            return null;
        }
        
        const word = line.substring(wordRange.start, wordRange.end);
        
        // Look for definitions in the document
        for (let i = 0; i < lines.length; i++) {
            const currentLine = lines[i];
            if (currentLine.includes(`node ${word}`) || 
                currentLine.includes(`method ${word}`) || 
                currentLine.includes(`kernel ${word}`) ||
                currentLine.includes(`parameter ${word}`)) {
                return {
                    uri: params.textDocument.uri,
                    range: {
                        start: { line: i, character: 0 },
                        end: { line: i, character: currentLine.length }
                    }
                };
            }
        }
        
        return null;
    }
);

connection.onDocumentSymbol(
    (params: DocumentSymbolParams): DocumentSymbol[] => {
        const document = documents.get(params.textDocument.uri);
        if (!document) {
            return [];
        }

        const symbols: DocumentSymbol[] = [];
        const text = document.getText();
        const lines = text.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Find node definitions
            const nodeMatch = line.match(/node\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
            if (nodeMatch) {
                symbols.push({
                    name: nodeMatch[1],
                    kind: SymbolKind.Class,
                    range: {
                        start: { line: i, character: 0 },
                        end: { line: i, character: line.length }
                    },
                    selectionRange: {
                        start: { line: i, character: line.indexOf(nodeMatch[1]) },
                        end: { line: i, character: line.indexOf(nodeMatch[1]) + nodeMatch[1].length }
                    }
                });
            }
            
            // Find method definitions
            const methodMatch = line.match(/method\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
            if (methodMatch) {
                symbols.push({
                    name: methodMatch[1],
                    kind: SymbolKind.Method,
                    range: {
                        start: { line: i, character: 0 },
                        end: { line: i, character: line.length }
                    },
                    selectionRange: {
                        start: { line: i, character: line.indexOf(methodMatch[1]) },
                        end: { line: i, character: line.indexOf(methodMatch[1]) + methodMatch[1].length }
                    }
                });
            }
            
            // Find kernel definitions
            const kernelMatch = line.match(/kernel\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
            if (kernelMatch) {
                symbols.push({
                    name: kernelMatch[1],
                    kind: SymbolKind.Function,
                    range: {
                        start: { line: i, character: 0 },
                        end: { line: i, character: line.length }
                    },
                    selectionRange: {
                        start: { line: i, character: line.indexOf(kernelMatch[1]) },
                        end: { line: i, character: line.indexOf(kernelMatch[1]) + kernelMatch[1].length }
                    }
                });
            }
        }
        
        return symbols;
    }
);

// Helper functions
function getWordRangeAtPosition(line: string, character: number): { start: number; end: number } | null {
    const wordRegex = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    let match;
    
    while ((match = wordRegex.exec(line)) !== null) {
        if (character >= match.index && character <= match.index + match[0].length) {
            return {
                start: match.index,
                end: match.index + match[0].length
            };
        }
    }
    
    return null;
}

function getKeywordDocumentation(keyword: string): string {
    const documentation: { [key: string]: string } = {
        'node': 'Defines a ROS2 node with its configuration and methods',
        'method': 'Defines a C++ method within a node',
        'kernel': 'Defines a CUDA kernel for GPU acceleration',
        'parameter': 'Defines a configurable parameter for the node',
        'timer': 'Defines a periodic timer for the node',
        'publisher': 'Defines a ROS2 publisher',
        'subscriber': 'Defines a ROS2 subscriber',
        'service': 'Defines a ROS2 service',
        'client': 'Defines a ROS2 service client',
        'action': 'Defines a ROS2 action',
        'include': 'Includes external files or libraries',
        'input': 'Defines an input parameter for a method or kernel',
        'output': 'Defines an output parameter for a method or kernel',
        'code': 'Contains C++ or CUDA code blocks',
        'qos': 'Quality of Service configuration for ROS topics',
        'cuda_kernels': 'Block containing CUDA kernel definitions',
        'onnx_model': 'Defines an ONNX model for machine learning inference',
        'pipeline': 'Defines a processing pipeline with multiple stages',
        'stage': 'Defines a stage within a pipeline'
    };
    
    return documentation[keyword] || `RoboDSL keyword: ${keyword}`;
}

function getHoverInformation(word: string): string | null {
    const hoverInfo: { [key: string]: string } = {
        'node': '**Node Definition**\n\nDefines a ROS2 node with its configuration, parameters, and methods.\n\n```robodsl\nnode MyNode {\n  parameter int rate: 10\n  method process {\n    input: sensor_msgs/Image image\n    code: {\n      // C++ code here\n    }\n  }\n}\n```',
        'method': '**Method Definition**\n\nDefines a C++ method within a node that processes data.\n\n```robodsl\nmethod process {\n  input: sensor_msgs/Image image\n  output: std_msgs/String result\n  code: {\n    // C++ implementation\n  }\n}\n```',
        'kernel': '**CUDA Kernel**\n\nDefines a CUDA kernel for GPU-accelerated processing.\n\n```robodsl\nkernel process_gpu {\n  input: float* data, int size\n  block_size: (256, 1, 1)\n  code: {\n    // CUDA kernel code\n  }\n}\n```',
        'publisher': '**ROS2 Publisher**\n\nDefines a publisher that sends messages to a topic.\n\n```robodsl\npublisher /output_topic: "std_msgs/String"\n```',
        'subscriber': '**ROS2 Subscriber**\n\nDefines a subscriber that receives messages from a topic.\n\n```robodsl\nsubscriber /input_topic: "sensor_msgs/Image"\n```',
        'parameter': '**Node Parameter**\n\nDefines a configurable parameter for the node.\n\n```robodsl\nparameter int rate: 10\nparameter string model_path: "/path/to/model"\n```',
        'timer': '**Timer**\n\nDefines a periodic timer that triggers at specified intervals.\n\n```robodsl\ntimer process_timer: 100ms\n```',
        'qos': '**Quality of Service**\n\nConfigures the quality of service for ROS topics.\n\n```robodsl\nqos {\n  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 10\n}\n```'
    };
    
    return hoverInfo[word] || null;
}

// Make the text document manager listen on the connection
// for open, change and close text document events
documents.listen(connection);

// Listen on the connection
connection.listen(); 