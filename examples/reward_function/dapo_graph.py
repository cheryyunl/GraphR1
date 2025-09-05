# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
from typing import Any, Dict, List

# Valid values from the dataset
VALID_ACTION_TYPES = {"press", "rotate", "pull", "open", "push", "close", "insert"}
VALID_FUNCTIONAL_RELATIONSHIPS = {"openorclose", "adjust", "control", "providepower", "activate"}
VALID_SPATIAL_RELATIONS = {"left_of", "right_of", "in_front_of", "behind", "higher_than", "lower_than", "close", "far", "touching"}

def extract_answer_json(response: str) -> Dict:
    """Extract JSON from Answer: field in response."""
    match = re.search(r"Answer\s*:\s*(\{.*\})", response, re.DOTALL)
    if not match:
        return None
    
    try:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def format_reward(response: str) -> float:
    """Check if prediction follows correct format with valid JSON structure."""
    # 1. Check basic Answer: format
    if not re.search(r"Answer\s*:\s*\{", response):
        return 0.0
    
    # 2. Extract and parse JSON
    answer_json = extract_answer_json(response)
    if answer_json is None:
        return 0.0
    
    # 3. Check required fields
    required_fields = ["task_instruction", "nodes", "edges", "action_type", "function_type"]
    if not all(field in answer_json for field in required_fields):
        return 0.0
    
    # 4. Check data types
    if not isinstance(answer_json.get("nodes"), list):
        return 0.0
    if not isinstance(answer_json.get("edges"), list):
        return 0.0
    if not isinstance(answer_json.get("task_instruction"), str):
        return 0.0
    
    # 5. Check valid enum values
    if answer_json.get("action_type") not in VALID_ACTION_TYPES:
        return 0.0
    
    # 6. Check edges structure
    for edge in answer_json.get("edges", []):
        if not isinstance(edge, dict):
            return 0.0
        
        required_edge_fields = ["functional_relationship", "object1", "object2", "spatial_relations", "is_touching"]
        if not all(field in edge for field in required_edge_fields):
            return 0.0
        
        # Check functional relationship validity
        if edge.get("functional_relationship") not in VALID_FUNCTIONAL_RELATIONSHIPS:
            return 0.0
        
        # Check spatial relations
        spatial_rels = edge.get("spatial_relations", [])
        if not isinstance(spatial_rels, list):
            return 0.0
        
        for rel in spatial_rels:
            if rel not in VALID_SPATIAL_RELATIONS:
                return 0.0
        
        # Check is_touching is boolean
        if not isinstance(edge.get("is_touching"), bool):
            return 0.0
    
    return 1.0

def calculate_graph_similarity(pred_json: Dict, gt_json: Dict) -> float:
    """Calculate similarity between predicted and ground truth graph structures."""
    if pred_json is None or gt_json is None:
        return 0.0
    
    total_score = 0.0
    components = 0
    
    # 1. Task instruction similarity (exact match)
    components += 1
    if pred_json.get("task_instruction") == gt_json.get("task_instruction"):
        total_score += 1.0
    
    # 2. Action type similarity (exact match) - CRITICAL
    components += 1
    if pred_json.get("action_type") == gt_json.get("action_type"):
        total_score += 1.0
    
    # 3. Function type similarity (exact match)
    components += 1
    if pred_json.get("function_type") == gt_json.get("function_type"):
        total_score += 1.0
    
    # 4. Nodes similarity (penalize extra nodes)
    components += 1
    pred_nodes = set(pred_json.get("nodes", []))
    gt_nodes = set(gt_json.get("nodes", []))
    if gt_nodes:
        # Penalize both missing and extra nodes
        intersection = len(pred_nodes.intersection(gt_nodes))
        union = len(pred_nodes.union(gt_nodes))
        nodes_similarity = intersection / union if union > 0 else 0.0
        total_score += nodes_similarity
    
    # 5. Edges similarity (more strict matching)
    components += 1
    pred_edges = pred_json.get("edges", [])
    gt_edges = gt_json.get("edges", [])
    
    if gt_edges:
        edge_score = 0.0
        for gt_edge in gt_edges:
            best_match = 0.0
            for pred_edge in pred_edges:
                edge_sim = calculate_edge_similarity(pred_edge, gt_edge)
                best_match = max(best_match, edge_sim)
            edge_score += best_match
        edge_score /= len(gt_edges)
        
        # Penalize extra edges
        if len(pred_edges) > len(gt_edges):
            penalty = (len(pred_edges) - len(gt_edges)) * 0.1
            edge_score = max(0, edge_score - penalty)
        
        total_score += edge_score
    
    return total_score / components

def calculate_edge_similarity(pred_edge: Dict, gt_edge: Dict) -> float:
    """Calculate similarity between two edges."""
    score = 0.0
    components = 0
    
    # Object matching (bidirectional)
    components += 1
    pred_objects = {pred_edge.get("object1"), pred_edge.get("object2")}
    gt_objects = {gt_edge.get("object1"), gt_edge.get("object2")}
    if pred_objects == gt_objects:
        score += 1.0
    
    # Functional relationship
    components += 1
    if pred_edge.get("functional_relationship") == gt_edge.get("functional_relationship"):
        score += 1.0
    
    # Spatial relations (strict matching with penalty for wrong relations)
    components += 1
    pred_spatial = set(pred_edge.get("spatial_relations", []))
    gt_spatial = set(gt_edge.get("spatial_relations", []))
    if gt_spatial:
        # Intersection over union to penalize both missing and extra relations
        intersection = len(pred_spatial.intersection(gt_spatial))
        union = len(pred_spatial.union(gt_spatial))
        spatial_sim = intersection / union if union > 0 else 0.0
        score += spatial_sim
    
    # Is touching
    components += 1
    if pred_edge.get("is_touching") == gt_edge.get("is_touching"):
        score += 1.0
    
    return score / components

def accuracy_reward(response: str, ground_truth: str) -> float:
    """Calculate accuracy reward by comparing predicted and ground truth graphs."""
    # Extract JSON from response
    pred_json = extract_answer_json(response)
    
    # Parse ground truth JSON
    try:
        gt_json = json.loads(ground_truth.strip())
    except json.JSONDecodeError:
        return -0.5
    
    # If failed to extract JSON, return negative reward
    if pred_json is None:
        return -0.5
    
    # If format validation fails (invalid enum values etc), give 0 accuracy
    if format_reward(response) == 0.0:
        return 0.0
    
    # Calculate similarity
    similarity = calculate_graph_similarity(pred_json, gt_json)
    
    # Convert to DAPO-style reward with finer granularity
    if similarity >= 0.98:  # Very strict for perfect score
        return 1.0
    elif similarity >= 0.85:
        return 0.8
    elif similarity >= 0.7:
        return 0.5
    elif similarity >= 0.5:
        return 0.2
    elif similarity >= 0.3:
        return 0.0
    else:
        return -0.5

def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int) -> float:
    """Apply soft length penalty for overly long responses."""
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0

def compute_score(
    reward_inputs: list[dict[str, Any]],
    max_response_length: int,
    overlong_buffer_length: int,
    overlong_penalty_factor: float,
    format_weight: float = 0.2,
) -> list[dict[str, float]]:
    """Compute DAPO-style scores for graph generation task."""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for dapo_graph reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        
        # Format validation (JSON structure check)
        format_score = format_reward(response)
        
        # Content accuracy (graph structure similarity)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # Length control
        overlong_score = soft_overlong_punishment(
            len(response), max_response_length, overlong_buffer_length
        )
        
        # Combined score: format_weight=0.2, accuracy_weight=0.8
        overall = (
            format_weight * format_score + 
            (1 - format_weight) * accuracy_score + 
            overlong_score * overlong_penalty_factor
        )
        
        scores.append({
            "overall": overall,
            "format": format_score,
            "accuracy": accuracy_score,
            "overlong": overlong_score,
            "accuracy_normalized": 0.5 * (accuracy_score + 1.0),  # For filtering
        })

    return scores
