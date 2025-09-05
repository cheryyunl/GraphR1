#!/usr/bin/env python3
"""
检查新数据结构
"""
import os
import json
import glob

def inspect_new_data():
    print("🔍 Inspecting new scene data structure...")
    
    # 检查新数据路径
    new_base = "/code/new_scene"
    if not os.path.exists(new_base):
        print(f"❌ Path not found: {new_base}")
        return
    
    print(f"📁 Exploring {new_base}...")
    
    # 递归查找所有JSON文件
    json_files = glob.glob(os.path.join(new_base, "**/*.json"), recursive=True)
    print(f"📄 Found {len(json_files)} JSON files")
    
    # 显示文件结构
    for i, json_file in enumerate(json_files[:5]):  # 只看前5个
        print(f"\n📋 File {i+1}: {json_file}")
        
        # 检查同级目录是否有rgb文件夹
        json_dir = os.path.dirname(json_file)
        rgb_dir = os.path.join(json_dir, "rgb")
        has_rgb = os.path.exists(rgb_dir)
        print(f"   RGB folder: {'✅' if has_rgb else '❌'} {rgb_dir}")
        
        if has_rgb:
            rgb_files = glob.glob(os.path.join(rgb_dir, "*"))
            print(f"   RGB files: {len(rgb_files)}")
        
        # 读取JSON内容
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   JSON type: {type(data)}")
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
                if 'task_instruction' in data:
                    print(f"   Task: {data['task_instruction'][:100]}...")
            elif isinstance(data, list) and len(data) > 0:
                print(f"   List length: {len(data)}")
                print(f"   First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not dict'}")
        except Exception as e:
            print(f"   ❌ Error reading JSON: {e}")
    
    # 检查目录结构模式
    print(f"\n📊 Directory patterns:")
    dirs = set()
    for json_file in json_files:
        rel_path = os.path.relpath(json_file, new_base)
        dir_parts = rel_path.split(os.sep)[:-1]  # 去掉文件名
        if len(dir_parts) >= 2:
            pattern = "/".join(dir_parts[:3])  # 取前3层
            dirs.add(pattern)
    
    for pattern in sorted(dirs)[:10]:  # 显示前10个模式
        print(f"   {pattern}/")

if __name__ == "__main__":
    inspect_new_data()
