#!/usr/bin/env python3
"""
测试斜杠分隔对象名称的匹配
"""
import sys
import os
sys.path.append('/Users/cheryunl/GraphR1/examples/reward_function')

from dapo_graph import objects_match, calculate_graph_similarity

def test_slash_object_matching():
    print("🧪 Testing slash-separated object matching...")
    
    # 测试用例
    test_cases = [
        # (predicted, ground_truth, should_match)
        ("faucet", "faucet / handle", True),
        ("handle", "faucet / handle", True),
        ("faucet", "faucet / handle", True),
        ("kitchen sink", "kitchen sink", True),
        ("sink", "kitchen sink", False),
        ("door", "faucet / handle", False),
        ("faucet / handle", "faucet", True),
        ("faucet / handle", "handle", True),
        ("faucet / handle", "faucet / handle", True),
    ]
    
    for pred, gt, expected in test_cases:
        result = objects_match(pred, gt)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{pred}' vs '{gt}': {result} (expected: {expected})")
    
    print("\n🧪 Testing full graph similarity with slash objects...")
    
    # 测试完整图相似度
    predicted_response = """
    Answer: {"task_instruction": "Control water flow of the kitchen sink.", "nodes": ["faucet", "kitchen sink"], "edges": [{"functional_relationship": "control", "object1": "faucet", "object2": "kitchen sink", "spatial_relations": ["higher_than", "behind", "close"], "is_touching": true}], "action_type": "open", "function_type": "water_flow_control"}
    """
    
    gt_json = {
        "task_instruction": "Control water flow of the kitchen sink.", 
        "nodes": ["faucet / handle", "kitchen sink"], 
        "edges": [{
            "functional_relationship": "control", 
            "object1": "faucet / handle", 
            "object2": "kitchen sink", 
            "spatial_relations": ["higher_than", "behind", "close"], 
            "is_touching": True
        }], 
        "action_type": "open", 
        "function_type": "water_flow_control"
    }
    
    # 导入必要的函数
    from dapo_graph import extract_answer_json, calculate_graph_similarity, format_reward, accuracy_reward
    
    # 测试
    pred_json = extract_answer_json(predicted_response)
    if pred_json:
        similarity = calculate_graph_similarity(pred_json, gt_json)
        format_score = format_reward(predicted_response)  # format_reward需要原始响应
        accuracy_score = accuracy_reward(similarity, format_score)
        
        print(f"📊 Results:")
        print(f"   Predicted JSON: {pred_json}")
        print(f"   Ground Truth: {gt_json}")
        print(f"   Graph Similarity: {similarity:.3f}")
        print(f"   Format Score: {format_score:.3f}")
        print(f"   Accuracy Score: {accuracy_score:.3f}")
        
        # 应该得到高分，因为只有节点名称不同
        if accuracy_score >= 0.8:
            print("✅ Test passed! Slash objects handled correctly.")
        else:
            print("❌ Test failed! Slash objects not handled properly.")
    else:
        print("❌ Failed to extract JSON from response")

if __name__ == "__main__":
    test_slash_object_matching()
