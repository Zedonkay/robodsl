from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from pygls.server import LanguageServer
from pygls.lsp.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionOptions,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeConfigurationParams,
    Hover,
    InitializeParams,
    Location,
    Position,
    Range,
    SymbolInformation,
    SymbolKind,
    TextDocumentContentChangeEvent,
    TextDocumentIdentifier,
    TextDocumentItem,
    TextEdit,
    WorkspaceEdit,
    SemanticTokens,
    FoldingRange,
    FoldingRangeKind,
    DocumentLink,
    CodeAction,
    CodeActionKind,
)

# Import parser/validator from project, with fallback sys.path adjustment
try:
    from robodsl.parsers.lark_parser import RoboDSLParser
    from robodsl.core.ast import RoboDSLAST, NodeNode
    from robodsl.core.validator import RoboDSLValidator, ValidationIssue, ValidationLevel
except Exception:
    # Add repo src directory to sys.path if running from source tree
    here = Path(__file__).resolve()
    for p in here.parents:
        src_dir = p / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
            break
    from robodsl.parsers.lark_parser import RoboDSLParser
    from robodsl.core.ast import RoboDSLAST, NodeNode
    from robodsl.core.validator import RoboDSLValidator, ValidationIssue, ValidationLevel


ROBO_DSL_LANGUAGE_ID = "robodsl"
ROBO_DSL_FILE_EXT = ".robodsl"


class RoboDSLLanguageServer(LanguageServer):
    CMD_FORMAT = "robodsl.format"
    CMD_VALIDATE = "robodsl.validate"

    def __init__(self):
        super().__init__("robodsl-ls", "0.1.0")
        self.parser = RoboDSLParser(debug=False)
        self.validator = RoboDSLValidator()
        # Simple workspace index: aggregate symbol names across files
        self.index_nodes: set[str] = set()
        self.index_kernels: set[str] = set()
        self.index_models: set[str] = set()


ls = RoboDSLLanguageServer()


def _text_to_lines(text: str) -> List[str]:
    return text.splitlines()


def _line_col_to_offset(lines: List[str], line: int, col: int) -> int:
    if line < 0:
        return 0
    if line >= len(lines):
        line = len(lines) - 1
        col = len(lines[line]) if line >= 0 else 0
    return sum(len(l) + 1 for l in lines[:line]) + min(col, len(lines[line]))


def _range_full_text(text: str) -> Range:
    lines = _text_to_lines(text)
    end_line = max(0, len(lines) - 1)
    end_char = len(lines[-1]) if lines else 0
    return Range(start=Position(line=0, character=0), end=Position(line=end_line, character=end_char))


def _issue_to_diagnostic(issue: ValidationIssue, text: str) -> Diagnostic:
    # Prefer line-based range if available
    if getattr(issue, "line", None):
        lines = _text_to_lines(text)
        line_index = max(0, int(issue.line) - 1)
        line_text = lines[line_index] if 0 <= line_index < len(lines) else ""
        start_char = 0 if getattr(issue, "column", None) is None else max(0, int(issue.column) - 1)
        end_char = len(line_text)
        rng = Range(start=Position(line=line_index, character=start_char), end=Position(line=line_index, character=end_char))
    else:
        # Fallback: whole document
        rng = _range_full_text(text)
    severity = DiagnosticSeverity.Error if issue.level == ValidationLevel.ERROR else DiagnosticSeverity.Warning
    return Diagnostic(range=rng, message=issue.message, severity=severity, source="robodsl")


@ls.feature("initialize")
def on_initialize(params: InitializeParams):
    return {
        "capabilities": {
            "textDocumentSync": 2,  # Incremental
            "documentSymbolProvider": True,
            "hoverProvider": True,
            "completionProvider": CompletionOptions(resolve_provider=False, trigger_characters=[":", " ", "/", "_", "<", ">", ".", "(", ","]),
            "documentFormattingProvider": True,
            "renameProvider": True,
            "workspaceSymbolProvider": True,
            "semanticTokensProvider": {
                "legend": {
                    "tokenTypes": [
                        "keyword",
                        "type",
                        "function",
                        "variable",
                        "string",
                        "number",
                    ],
                    "tokenModifiers": ["declaration", "readonly"],
                },
                "range": True,
                "full": True,
            },
            "foldingRangeProvider": True,
            "documentLinkProvider": {"resolveProvider": False},
            "codeActionProvider": {"codeActionKinds": [CodeActionKind.QuickFix, CodeActionKind.Refactor]},
        }
    }


@ls.feature("textDocument/didOpen")
def did_open(params):
    text_doc: TextDocumentItem = params.textDocument
    _publish_diagnostics(text_doc.uri, text_doc.text)


@ls.feature("textDocument/didChange")
def did_change(params):
    doc = ls.workspace.get_document(params.textDocument.uri)
    _publish_diagnostics(doc.uri, doc.source)


@ls.feature("textDocument/didSave")
def did_save(params):
    doc = ls.workspace.get_document(params.textDocument.uri)
    _publish_diagnostics(doc.uri, doc.source)


def _publish_diagnostics(uri: str, text: str) -> None:
    issues = ls.validator.validate_string(text)
    diags = [_issue_to_diagnostic(i, text) for i in issues]
    ls.publish_diagnostics(uri, diags)


@ls.feature("textDocument/documentSymbol")
def document_symbol(params) -> List[SymbolInformation]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    text = doc.source
    symbols: List[SymbolInformation] = []
    try:
        for name, kind, rng in _extract_top_level_symbols_with_ranges(text):
            symbols.append(SymbolInformation(name=name, kind=kind, location=Location(uri=doc.uri, range=rng)))
    except Exception:
        pass
    return symbols


@ls.feature("textDocument/hover")
def hover(params) -> Optional[Hover]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    text = doc.source
    word = _word_at_position(text, params.position)
    if not word:
        return None
    context = _symbol_context_at_position(text, params.position)
    contents = f"{context}: {word}" if context else f"RoboDSL symbol: {word}"
    return Hover(contents=contents)


def _word_at_position(text: str, pos: Position) -> Optional[str]:
    lines = _text_to_lines(text)
    if pos.line < 0 or pos.line >= len(lines):
        return None
    line_text = lines[pos.line]
    col = min(pos.character, len(line_text))
    start = col
    while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] in "_/:"):
        start -= 1
    end = col
    while end < len(line_text) and (line_text[end].isalnum() or line_text[end] in "_/:"):
        end += 1
    word = line_text[start:end].strip()
    return word or None


@ls.feature("textDocument/completion")
def completion(params) -> CompletionList:
    doc = ls.workspace.get_document(params.textDocument.uri)
    text_before = _text_before_position(doc.source, params.position)
    items: List[CompletionItem] = []

    # Keywords from grammar
    keywords = _get_grammar_keywords()
    for kw in keywords:
        items.append(CompletionItem(label=kw, kind=CompletionItemKind.Keyword))

    # Contextual suggestions
    if text_before.rstrip().endswith("publisher") or ":" in text_before and "publisher" in text_before:
        items.extend([
            CompletionItem(label="qos", kind=CompletionItemKind.Property),
            CompletionItem(label="queue_size", kind=CompletionItemKind.Property),
            CompletionItem(label="history", kind=CompletionItemKind.Property),
            CompletionItem(label="depth", kind=CompletionItemKind.Property),
        ])

    return CompletionList(is_incomplete=False, items=items)


def _text_before_position(text: str, pos: Position) -> str:
    lines = _text_to_lines(text)
    if pos.line < 0 or pos.line >= len(lines):
        return ""
    return "\n".join(lines[: pos.line] + [lines[pos.line][: pos.character]])


@ls.feature("textDocument/formatting")
def formatting(params) -> List[TextEdit]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    formatted = ls.validator.linter.format_string(doc.source) if hasattr(ls.validator, "linter") else None
    if not formatted:
        try:
            # Fallback to validator's linter
            from robodsl.core.validator import RoboDSLLinter

            linter = RoboDSLLinter()
            formatted = linter.format_string(doc.source)
        except Exception:
            formatted = doc.source
    if formatted == doc.source:
        return []
    return [TextEdit(range=_range_full_text(doc.source), new_text=formatted)]


@ls.feature("textDocument/rename")
def rename(params) -> Optional[WorkspaceEdit]:
    # Best-effort whole word rename in current document only
    doc = ls.workspace.get_document(params.textDocument.uri)
    text = doc.source
    word = _word_at_position(text, params.position)
    if not word:
        return None
    new_name = params.newName
    new_text = text.replace(word, new_name)
    return WorkspaceEdit(changes={doc.uri: [TextEdit(range=_range_full_text(text), new_text=new_text)]})


@ls.feature("workspace/symbol")
def workspace_symbol(params) -> List[SymbolInformation]:
    results: List[SymbolInformation] = []
    try:
        root = ls.workspace.root_path
        if not root:
            return []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(ROBO_DSL_FILE_EXT):
                    path = os.path.join(dirpath, fn)
                    try:
                        text = Path(path).read_text(encoding="utf-8")
                    except Exception:
                        continue
                    for name, kind, rng in _extract_top_level_symbols_with_ranges(text):
                        results.append(SymbolInformation(name=name, kind=kind, location=Location(uri=Path(path).as_uri(), range=rng)))
        return results
    except Exception:
        return []


@ls.command(RoboDSLLanguageServer.CMD_VALIDATE)
def cmd_validate(ls: RoboDSLLanguageServer, *args):
    # No-op command; validation runs on changes
    pass


@ls.feature("textDocument/semanticTokens/full")
def semantic_tokens_full(params) -> SemanticTokens:
    doc = ls.workspace.get_document(params.textDocument.uri)
    data = _compute_semantic_tokens(doc.source)
    return SemanticTokens(data=data)


@ls.feature("textDocument/semanticTokens/range")
def semantic_tokens_range(params) -> SemanticTokens:
    doc = ls.workspace.get_document(params.textDocument.uri)
    # For simplicity, compute full and rely on client to clip
    data = _compute_semantic_tokens(doc.source)
    return SemanticTokens(data=data)


@ls.feature("textDocument/foldingRange")
def folding_range(params) -> List[FoldingRange]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    return _compute_folding_ranges(doc.source)


@ls.feature("textDocument/documentLink")
def document_link(params) -> List[DocumentLink]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    return _compute_document_links(doc.source, doc.uri)


@ls.feature("textDocument/codeAction")
def code_action(params) -> List[CodeAction]:
    doc = ls.workspace.get_document(params.textDocument.uri)
    diagnostics = params.context.diagnostics or []
    edits: List[CodeAction] = []
    lines = _text_to_lines(doc.source)
    # Quick fix: add missing leading slash to topics
    for i, line in enumerate(lines):
        if "publisher" in line or "subscriber" in line or "service" in line:
            if ":" in line:
                parts = line.split(":", 1)
                right = parts[1].strip()
                if right and right[0] == '"' and not right.startswith("\"/"):
                    new_line = parts[0] + ": \"/" + right.strip('"') + "\""
                    edits.append(CodeAction(
                        title="Prefix topic with /",
                        kind=CodeActionKind.QuickFix,
                        edit=WorkspaceEdit(changes={doc.uri: [TextEdit(range=Range(start=Position(i, 0), end=Position(i, len(line))), new_text=new_line)]}),
                    ))
    # Quick fix: add newline at EOF
    if doc.source and not doc.source.endswith("\n"):
        edits.append(CodeAction(
            title="Add newline at end of file",
            kind=CodeActionKind.QuickFix,
            edit=WorkspaceEdit(changes={doc.uri: [TextEdit(range=_range_full_text(doc.source), new_text=doc.source + "\n")]})
        ))
    return edits


def run():
    ls.start_io()


# ------------------------
# Helpers for symbols/ranges/sem tokens
# ------------------------

_GRAMMAR_KEYWORDS_CACHE: Optional[List[str]] = None


def _get_grammar_keywords() -> List[str]:
    global _GRAMMAR_KEYWORDS_CACHE
    if _GRAMMAR_KEYWORDS_CACHE is not None:
        return _GRAMMAR_KEYWORDS_CACHE
    # Try to read grammar file to extract keywords
    grammar_paths = []
    # From installed package src layout
    try:
        from importlib.resources import files

        grammar_candidate = files("robodsl").joinpath("grammar/robodsl.lark")
        if grammar_candidate and grammar_candidate.is_file():
            grammar_paths.append(str(grammar_candidate))
    except Exception:
        pass
    # From source tree
    here = Path(__file__).resolve()
    for p in here.parents:
        candidate = p / "src" / "robodsl" / "grammar" / "robodsl.lark"
        if candidate.exists():
            grammar_paths.append(str(candidate))
            break
    keywords: List[str] = []
    for gp in grammar_paths:
        try:
            content = Path(gp).read_text(encoding="utf-8")
        except Exception:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Match TOKEN: "keyword"
            if ":" in line and '"' in line and line.split(":", 1)[0].isupper():
                try:
                    literal = line.split(":", 1)[1].strip()
                    if literal.startswith('"') and '"' in literal[1:]:
                        kw = literal.split('"')[1]
                        if kw.isidentifier() or kw in {"cpp:", "code:", "cuda_kernels", "kernel", "onnx_model", "pipeline", "node"}:
                            keywords.append(kw)
                except Exception:
                    continue
    if not keywords:
        keywords = [
            "package",
            "node",
            "parameter",
            "publisher",
            "subscriber",
            "service",
            "action",
            "client",
            "timer",
            "remap",
            "namespace",
            "cpp:",
            "code:",
            "cuda_kernels",
            "kernel",
            "onnx_model",
            "pipeline",
        ]
    _GRAMMAR_KEYWORDS_CACHE = sorted(set(keywords))
    return _GRAMMAR_KEYWORDS_CACHE


def _extract_top_level_symbols_with_ranges(text: str) -> List[Tuple[str, int, Range]]:
    lines = _text_to_lines(text)
    results: List[Tuple[str, int, Range]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # node NAME { ... }
        for prefix, kind in (("node ", SymbolKind.Class), ("kernel ", SymbolKind.Function), ("onnx_model ", SymbolKind.Interface)):
            if stripped.startswith(prefix):
                # Extract name
                try:
                    name = stripped[len(prefix) :].split()[0]
                except Exception:
                    name = ""
                # Find block range by brace balance
                start_line = i
                brace_balance = line.count("{") - line.count("}")
                j = i
                while j + 1 < len(lines) and brace_balance > 0:
                    j += 1
                    brace_balance += lines[j].count("{") - lines[j].count("}")
                end_line = j
                rng = Range(start=Position(start_line, 0), end=Position(end_line, len(lines[end_line])))
                if name:
                    results.append((name, kind, rng))
                break
        i += 1
    return results


def _symbol_context_at_position(text: str, pos: Position) -> Optional[str]:
    lines = _text_to_lines(text)
    # Walk upward to find nearest header
    for l in range(pos.line, -1, -1):
        s = lines[l].lstrip()
        if s.startswith("node "):
            return "Node"
        if s.startswith("kernel "):
            return "CUDA Kernel"
        if s.startswith("onnx_model "):
            return "ONNX Model"
        if s.startswith("publisher"):
            return "Publisher"
        if s.startswith("subscriber"):
            return "Subscriber"
        if s.startswith("service "):
            return "Service"
        if s.startswith("action "):
            return "Action"
    return None


def _compute_semantic_tokens(text: str) -> List[int]:
    token_types = ["keyword", "type", "function", "variable", "string", "number"]
    type_index = {t: i for i, t in enumerate(token_types)}
    keywords = set(_get_grammar_keywords())
    data: List[int] = []
    prev_line = 0
    prev_start = 0
    lines = _text_to_lines(text)
    import re

    string_pattern = re.compile(r'"([^"\\]|\\.)*"')
    number_pattern = re.compile(r"-?\b\d+(?:\.\d+)?\b")
    ident_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*:?")

    for line_idx, line in enumerate(lines):
        # strings
        for m in string_pattern.finditer(line):
            start, end = m.span()
            data.extend([
                line_idx - prev_line,
                start if line_idx != prev_line else start - prev_start,
                end - start,
                type_index["string"],
                0,
            ])
            prev_line, prev_start = line_idx, start
        # numbers
        for m in number_pattern.finditer(line):
            start, end = m.span()
            data.extend([
                line_idx - prev_line,
                start if line_idx != prev_line else start - prev_start,
                end - start,
                type_index["number"],
                0,
            ])
            prev_line, prev_start = line_idx, start
        # keywords and simple identifiers
        for m in ident_pattern.finditer(line):
            token = m.group(0)
            if token in keywords:
                start, end = m.span()
                data.extend([
                    line_idx - prev_line,
                    start if line_idx != prev_line else start - prev_start,
                    end - start,
                    type_index["keyword"],
                    0,
                ])
                prev_line, prev_start = line_idx, start
    return data


def _compute_folding_ranges(text: str) -> List[FoldingRange]:
    lines = _text_to_lines(text)
    stack: List[int] = []
    ranges: List[FoldingRange] = []
    for i, line in enumerate(lines):
        if "{" in line:
            stack.append(i)
        if "}" in line and stack:
            start = stack.pop()
            if i > start:
                ranges.append(FoldingRange(startLine=start, endLine=i, kind=FoldingRangeKind.Region))
    return ranges


def _compute_document_links(text: str, uri: str) -> List[DocumentLink]:
    lines = _text_to_lines(text)
    links: List[DocumentLink] = []
    import re
    # include "path" or include <path>
    inc_q = re.compile(r'include\s+"([^"]+)"')
    inc_a = re.compile(r'include\s+<([^>]+)>')
    for i, line in enumerate(lines):
        for m in inc_q.finditer(line):
            start, end = m.span(1)
            target = m.group(1)
            try:
                base = Path(uri.replace("file://", ""))
                resolved = (base.parent / target).resolve().as_uri()
                links.append(DocumentLink(range=Range(start=Position(i, start), end=Position(i, end)), target=resolved))
            except Exception:
                continue
        for m in inc_a.finditer(line):
            start, end = m.span(1)
            target = m.group(1)
            # System includes likely not resolvable; skip
            continue
    return links


