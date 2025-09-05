#!/usr/bin/env python3
"""Test script for dapo_graph reward function."""

import json
import sys
import os

# Add the reward_function directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reward_function'))

from dapo_graph import compute_score

def test_graph_reward():
    """Test the dapo_graph reward function with various scenarios."""
    
    # Ground truth JSON (from example_graph.py, but corrected format)
    gt_json = {
        "task_instruction": "Power on the toaster.",
        "nodes": ["electric outlet", "toaster"],
        "edges": [
            {
                "functional_relationship": "providepower",
                "object1": "electric outlet",
                "object2": "toaster",
                "spatial_relations": ["higher_than", "right_of", "in_front_of", "close"],
                "is_touching": False
            }
        ],
        "action_type": "insert",
        "function_type": "power_supply"
    }
    ground_truth = json.dumps(gt_json)
    
    # Test cases
    test_cases = [
        {
            "name": "Perfect Match",
            "response": f"""First, I understand the task: "Power on the toaster."
Then, I identify the key objects needed for this task: an electric outlet and a toaster.
Next, I analyze their spatial relationships: the outlet is higher than, to the right of, in front of, and close to the toaster.
Finally, I determine their functional relationship: the outlet can provide power to the toaster.

Answer: {json.dumps(gt_json)}""",
            "expected_accuracy": 1.0,
            "expected_format": 1.0
        },
        
        {
            "name": "Wrong Action Type",
            "response": f"""I see an outlet and toaster that need to be connected.

Answer: {json.dumps({**gt_json, "action_type": "press"})}""",
            "expected_accuracy": 0.5,  # Should be lower due to wrong action
            "expected_format": 1.0
        },
        
        {
            "name": "Missing Required Field",
            "response": """I analyze the image and see objects.

Answer: {"task_instruction": "Power on the toaster.", "nodes": ["electric outlet", "toaster"]}""",
            "expected_accuracy": 0.0,
            "expected_format": 0.0  # Missing required fields
        },
        
        {
            "name": "Invalid JSON",
            "response": """I see the task requires powering on a toaster.

Answer: {invalid json format}""",
            "expected_accuracy": -0.5,  # Should get negative reward
            "expected_format": 0.0
        },
        
        {
            "name": "No Answer Field",
            "response": """I analyze the image and see an outlet and toaster. 
The outlet is higher than the toaster and they need to be connected.""",
            "expected_accuracy": -0.5,  # Should get negative reward  
            "expected_format": 0.0
        },
        
        {
            "name": "Extra Objects (should be penalized)",
            "response": f"""Answer: {json.dumps({
                **gt_json, 
                "nodes": ["electric outlet", "toaster", "kitchen counter", "window"]
            })}""",
            "expected_accuracy": 0.8,  # Slightly lower due to extra nodes
            "expected_format": 1.0
        },
        
        {
            "name": "Wrong Spatial Relations",
            "response": f"""Answer: {json.dumps({
                **gt_json,
                "edges": [{
                    **gt_json["edges"][0],
                    "spatial_relations": ["left_of", "far"]
                }]
            })}""",
            "expected_accuracy": 0.8,  # Should get 0.8 due to 0.95 similarity
            "expected_format": 1.0
        },
        
        {
            "name": "Invalid Enum Values",
            "response": f"""Answer: {json.dumps({
                **gt_json,
                "action_type": "invalid_action",
                "edges": [{
                    **gt_json["edges"][0],
                    "functional_relationship": "invalid_relationship"
                }]
            })}""",
            "expected_accuracy": 0.0,  # Accuracy should be 0 due to invalid values  
            "expected_format": 0.0  # Invalid enum values should fail format check
        },
        
        {
            "name": "Very Long Response (length penalty)",
            "response": f"""{"This is a very long analysis. " * 100}
Answer: {json.dumps(gt_json)}""",
            "expected_accuracy": 1.0,
            "expected_format": 1.0,
            "expect_length_penalty": True
        }
    ]
    
    print("Testing DAPO Graph Reward Function")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        # Prepare input
        reward_inputs = [{
            "response": test_case["response"],
            "ground_truth": ground_truth,
            "response_length": len(test_case["response"])
        }]
        
        # Compute scores
        scores = compute_score(reward_inputs)
        score = scores[0]
        
        # Print results
        print(f"Overall Score: {score['overall']:.3f}")
        print(f"Format Score: {score['format']:.3f}")
        print(f"Accuracy Score: {score['accuracy']:.3f}")
        print(f"Length Penalty: {score['overlong']:.3f}")
        print(f"Accuracy Normalized: {score['accuracy_normalized']:.3f}")
        
        # Validation
        format_ok = abs(score['format'] - test_case['expected_format']) < 0.1
        accuracy_ok = abs(score['accuracy'] - test_case['expected_accuracy']) < 0.3  # Allow some tolerance
        
        if test_case.get('expect_length_penalty'):
            length_ok = score['overlong'] < 0  # Should have negative penalty
        else:
            length_ok = score['overlong'] >= 0  # Should not be penalized
        
        status = "✓ PASS" if (format_ok and accuracy_ok and length_ok) else "✗ FAIL"
        print(f"Status: {status}")
        
        if not format_ok:
            print(f"  Format mismatch: expected {test_case['expected_format']}, got {score['format']}")
        if not accuracy_ok:
            print(f"  Accuracy mismatch: expected ~{test_case['expected_accuracy']}, got {score['accuracy']}")
        if not length_ok:
            print(f"  Length penalty unexpected: got {score['overlong']}")
            

def test_edge_cases():
    """Test additional edge cases."""
    print("\n" + "=" * 50)
    print("Testing Edge Cases")
    print("=" * 50)
    
    gt_json = {"task_instruction": "Test task", "nodes": ["obj1"], "edges": [], "action_type": "press", "function_type": "control"}
    ground_truth = json.dumps(gt_json)
    
    edge_cases = [
        {
            "name": "Empty Response",
            "response": "",
        },
        {
            "name": "Only Answer Field",
            "response": f"Answer: {json.dumps(gt_json)}",
        },
        {
            "name": "Multiple Answer Fields",
            "response": f"Answer: invalid\nAnswer: {json.dumps(gt_json)}",
        }
    ]
    
    for case in edge_cases:
        print(f"\nEdge Case: {case['name']}")
        reward_inputs = [{"response": case["response"], "ground_truth": ground_truth, "response_length": len(case["response"])}]
        scores = compute_score(reward_inputs)
        score = scores[0]
        print(f"Overall: {score['overall']:.3f}, Format: {score['format']:.3f}, Accuracy: {score['accuracy']:.3f}")

if __name__ == "__main__":
    test_graph_reward()
    test_edge_cases()
    print("\n" + "=" * 50)
    print("Testing Complete!")
