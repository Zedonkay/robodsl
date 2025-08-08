"""Pipeline Generator for RoboDSL.

This module generates ROS2 nodes for pipeline stages and creates
the necessary topic connections between stages.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from ..core.ast import PipelineNode, StageNode, StageContentNode
from .base_generator import BaseGenerator


class PipelineGenerator(BaseGenerator):
    """Generates pipeline code from AST."""
    
    def __init__(self, output_dir: str = "build"):
        super().__init__(output_dir)
        self.template_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent.parent / "templates" / "pipeline"),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate(self, pipeline: PipelineNode, project_name: str = "robodsl_project") -> Dict[str, str]:
        """Generate pipeline code from AST."""
        generated_files = {}
        
        # Generate a node for each stage
        for i, stage in enumerate(pipeline.content.stages):
            stage_files = self._generate_stage_node(stage, pipeline.name, i, project_name)
            generated_files.update(stage_files)
        
        # Generate pipeline launch file
        launch_file = self._generate_pipeline_launch(pipeline, project_name)
        generated_files[f"launch/{pipeline.name}_pipeline.launch.py"] = launch_file
        
        # Generate pipeline documentation
        doc_file = self._generate_pipeline_doc(pipeline)
        generated_files[f"docs/{pipeline.name}_pipeline.md"] = doc_file
        
        return generated_files
    
    def _generate_stage_node(self, stage: StageNode, pipeline_name: str, stage_index: int, project_name: str) -> Dict[str, str]:
        """Generate ROS2 node for a pipeline stage."""
        files = {}
        
        # Create stage-specific namespace
        stage_namespace = f"/{pipeline_name}/{stage.name}"
        
        # Generate C++ node header
        header_template = self.template_env.get_template("stage_node.hpp.jinja2")
        header_content = header_template.render(
            stage=stage,
            pipeline_name=pipeline_name,
            stage_namespace=stage_namespace,
            project_name=project_name,
            stage_index=stage_index,
            has_cuda=len(stage.content.cuda_kernels) > 0,
            has_onnx=len(stage.content.onnx_models) > 0
        )
        files[f"include/pipelines/{project_name}/{stage.name}_node.hpp"] = header_content
        
        # Generate C++ node implementation
        impl_template = self.template_env.get_template("stage_node.cpp.jinja2")
        impl_content = impl_template.render(
            stage=stage,
            pipeline_name=pipeline_name,
            stage_namespace=stage_namespace,
            project_name=project_name,
            stage_index=stage_index,
            has_cuda=len(stage.content.cuda_kernels) > 0,
            has_onnx=len(stage.content.onnx_models) > 0
        )
        files[f"src/pipelines/{stage.name}_node.cpp"] = impl_content
        
        # Python generation disabled - focusing on C++/CUDA only
        
        # Generate CUDA kernel integration if needed
        if stage.content.cuda_kernels:
            cuda_files = self._generate_cuda_integration(stage, project_name)
            files.update(cuda_files)
        
        # Generate ONNX integration if needed
        if stage.content.onnx_models:
            onnx_files = self._generate_onnx_integration(stage, project_name)
            files.update(onnx_files)
        
        return files
    
    def _generate_cuda_integration(self, stage: StageNode, project_name: str) -> Dict[str, str]:
        """Generate CUDA kernel integration files for a stage."""
        files = {}
        
        # Generate CUDA kernel wrapper header
        cuda_header_template = self.template_env.get_template("cuda_integration.hpp.jinja2")
        cuda_header_content = cuda_header_template.render(
            stage=stage,
            project_name=project_name
        )
        files[f"include/pipelines/{project_name}/{stage.name}_cuda.hpp"] = cuda_header_content
        
        # Generate CUDA kernel wrapper implementation
        cuda_impl_template = self.template_env.get_template("cuda_integration.cpp.jinja2")
        cuda_impl_content = cuda_impl_template.render(
            stage=stage,
            project_name=project_name
        )
        files[f"src/pipelines/{stage.name}_cuda.cpp"] = cuda_impl_content
        
        return files
    
    def _generate_onnx_integration(self, stage: StageNode, project_name: str) -> Dict[str, str]:
        """Generate ONNX model integration files for a stage."""
        files = {}
        
        # Generate ONNX integration header
        onnx_header_template = self.template_env.get_template("onnx_integration.hpp.jinja2")
        onnx_header_content = onnx_header_template.render(
            stage=stage,
            project_name=project_name
        )
        files[f"include/pipelines/{project_name}/{stage.name}_onnx.hpp"] = onnx_header_content
        
        # Generate ONNX integration implementation
        onnx_impl_template = self.template_env.get_template("onnx_integration.cpp.jinja2")
        onnx_impl_content = onnx_impl_template.render(
            stage=stage,
            project_name=project_name
        )
        files[f"src/pipelines/{stage.name}_onnx.cpp"] = onnx_impl_content
        
        return files
    
    def _generate_pipeline_launch(self, pipeline: PipelineNode, project_name: str) -> str:
        """Generate launch file for the entire pipeline."""
        launch_template = self.template_env.get_template("pipeline.launch.py.jinja2")
        return launch_template.render(
            pipeline=pipeline,
            project_name=project_name
        )
    
    def _generate_pipeline_doc(self, pipeline: PipelineNode) -> str:
        """Generate documentation for the pipeline."""
        doc_template = self.template_env.get_template("pipeline_doc.md.jinja2")
        return doc_template.render(pipeline=pipeline)
    
    def _get_stage_input_topics(self, stage: StageNode, pipeline_name: str) -> List[str]:
        """Get input topics for a stage."""
        topics = []
        for input_node in stage.content.inputs:
            # Create topic name based on input name
            topic_name = f"/{pipeline_name}/{input_node.input_name}"
            topics.append(topic_name)
        return topics
    
    def _get_stage_output_topics(self, stage: StageNode, pipeline_name: str) -> List[str]:
        """Get output topics for a stage."""
        topics = []
        for output_node in stage.content.outputs:
            # Create topic name based on output name
            topic_name = f"/{pipeline_name}/{output_node.output_name}"
            topics.append(topic_name)
        return topics
    
    def _get_stage_methods(self, stage: StageNode) -> List[str]:
        """Get methods for a stage."""
        return [method.method_name for method in stage.content.methods]
    
    def _get_stage_models(self, stage: StageNode) -> List[str]:
        """Get models for a stage."""
        return [model.model_name for model in stage.content.models]
    
    def _get_stage_topics(self, stage: StageNode) -> List[str]:
        """Get explicit topics for a stage."""
        return [topic.topic_path for topic in stage.content.topics] 