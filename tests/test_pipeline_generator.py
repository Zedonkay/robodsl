"""Tests for pipeline generator functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.robodsl.parsers.lark_parser import RoboDSLParser
from src.robodsl.generators.pipeline_generator import PipelineGenerator
from src.robodsl.core.ast import PipelineNode, StageNode, StageContentNode, PipelineContentNode, StageInputNode, StageOutputNode, StageMethodNode, StageModelNode, StageTopicNode


class TestPipelineGenerator:
    """Test pipeline generator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = PipelineGenerator(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_parsing(self):
        """Test that pipeline syntax is parsed correctly."""
        # Simple pipeline definition
        pipeline_text = """
        pipeline test_pipeline {
            stage stage1 {
                input: "input1"
                output: "output1"
                method: "process1"
            }
            
            stage stage2 {
                input: "output1"
                output: "output2"
                method: "process2"
                topic: "/test/topic"
            }
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(pipeline_text)
        
        # Verify pipeline was parsed
        assert len(ast.pipelines) == 1
        pipeline = ast.pipelines[0]
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.content.stages) == 2
        
        # Verify first stage
        stage1 = pipeline.content.stages[0]
        assert stage1.name == "stage1"
        assert len(stage1.content.inputs) == 1
        assert stage1.content.inputs[0].input_name == "input1"
        assert len(stage1.content.outputs) == 1
        assert stage1.content.outputs[0].output_name == "output1"
        assert len(stage1.content.methods) == 1
        assert stage1.content.methods[0].method_name == "process1"
        
        # Verify second stage
        stage2 = pipeline.content.stages[1]
        assert stage2.name == "stage2"
        assert len(stage2.content.inputs) == 1
        assert stage2.content.inputs[0].input_name == "output1"
        assert len(stage2.content.outputs) == 1
        assert stage2.content.outputs[0].output_name == "output2"
        assert len(stage2.content.methods) == 1
        assert stage2.content.methods[0].method_name == "process2"
        assert len(stage2.content.topics) == 1
        assert stage2.content.topics[0].topic_path == "/test/topic"
    
    def test_pipeline_generation(self):
        """Test that pipeline generates correct files."""
        # Create a simple pipeline AST
        pipeline = PipelineNode(
            name="test_pipeline",
            content=PipelineContentNode(
                stages=[
                    StageNode(
                        name="stage1",
                        content=StageContentNode(
                            inputs=[StageInputNode("input1")],
                            outputs=[StageOutputNode("output1")],
                            methods=[StageMethodNode("process1")]
                        )
                    ),
                    StageNode(
                        name="stage2",
                        content=StageContentNode(
                            inputs=[StageInputNode("output1")],
                            outputs=[StageOutputNode("output2")],
                            methods=[StageMethodNode("process2")],
                            topics=[StageTopicNode("/test/topic")]
                        )
                    )
                ]
            )
        )
        
        # Generate pipeline files
        generated_files = self.generator.generate(pipeline, "test_project")
        
        # Verify files were generated
        assert len(generated_files) > 0
        
        # Check for expected files
        expected_files = [
            "include/test_project/stage1_node.hpp",
            "src/stage1_node.cpp",
            "src/stage1_node.py",
            "include/test_project/stage2_node.hpp",
            "src/stage2_node.cpp",
            "src/stage2_node.py",
            "launch/test_pipeline_pipeline.launch.py",
            "docs/test_pipeline_pipeline.md"
        ]
        
        for expected_file in expected_files:
            assert expected_file in generated_files
        
        # Verify C++ header content
        header_content = generated_files["include/test_project/stage1_node.hpp"]
        assert "class Stage1Node" in header_content
        assert "on_input1_received" in header_content
        assert "output1_publisher_" in header_content
        
        # Verify launch file content
        launch_content = generated_files["launch/test_pipeline_pipeline.launch.py"]
        assert "test_pipeline" in launch_content
        assert "stage1_node" in launch_content
        assert "stage2_node" in launch_content
    
    def test_pipeline_with_models(self):
        """Test pipeline generation with ML models."""
        pipeline = PipelineNode(
            name="ml_pipeline",
            content=PipelineContentNode(
                stages=[
                    StageNode(
                        name="inference",
                        content=StageContentNode(
                            inputs=[StageInputNode("input_data")],
                            outputs=[StageOutputNode("predictions")],
                            methods=[StageMethodNode("infer")],
                            models=[StageModelNode("yolo_model")]
                        )
                    )
                ]
            )
        )
        
        generated_files = self.generator.generate(pipeline, "ml_project")
        
        # Verify model reference is included
        cpp_content = generated_files["src/inference_node.cpp"]
        assert "yolo_model" in cpp_content
        
        # Verify documentation includes model info
        doc_content = generated_files["docs/ml_pipeline_pipeline.md"]
        assert "yolo_model" in doc_content
    
    def test_complex_pipeline(self):
        """Test a more complex pipeline with multiple stages and configurations."""
        pipeline_text = """
        pipeline complex_pipeline {
            stage data_collection {
                input: "sensor_data"
                output: "raw_data"
                method: "collect"
                topic: "/sensors/raw"
            }
            
            stage preprocessing {
                input: "raw_data"
                output: "clean_data"
                method: "filter"
                method: "normalize"
                topic: "/data/clean"
            }
            
            stage analysis {
                input: "clean_data"
                output: "results"
                method: "analyze"
                model: "ml_model"
                topic: "/analysis/output"
            }
            
            stage visualization {
                input: "results"
                output: "plots"
                method: "plot"
                method: "save"
                topic: "/viz/plots"
            }
        }
        """
        
        parser = RoboDSLParser()
        ast = parser.parse(pipeline_text)
        
        assert len(ast.pipelines) == 1
        pipeline = ast.pipelines[0]
        assert pipeline.name == "complex_pipeline"
        assert len(pipeline.content.stages) == 4
        
        # Generate files
        generated_files = self.generator.generate(pipeline, "complex_project")
        
        # Verify all stages generated files
        stage_names = ["data_collection", "preprocessing", "analysis", "visualization"]
        for stage_name in stage_names:
            assert f"include/complex_project/{stage_name}_node.hpp" in generated_files
            assert f"src/{stage_name}_node.cpp" in generated_files
            assert f"src/{stage_name}_node.py" in generated_files
        
        # Verify launch file includes all stages
        launch_content = generated_files["launch/complex_pipeline_pipeline.launch.py"]
        for stage_name in stage_names:
            assert stage_name in launch_content


if __name__ == "__main__":
    pytest.main([__file__]) 