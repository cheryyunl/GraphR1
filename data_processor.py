#!/usr/bin/env python3
"""
Scene Graph Dataset Processor
将原始场景数据转换为HuggingFace数据集格式
"""

import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, DatasetDict
import random
from tqdm import tqdm
from collections import defaultdict

class SceneGraphProcessor:
    def __init__(self):
        # Functional relationship标准化映射
        self.func_rel_mapping = {
            "open or close": "openorclose",
            "openorclose": "openorclose",
            "adjust": "adjust", 
            "control": "control",
            "provide power": "providepower",
            "providepower": "providepower",
            "activate": "activate"
        }
    
    def find_all_scenes(self, base_folders):
        """递归查找所有场景文件夹"""
        all_scenes = []
        
        for base_folder in base_folders:
            if not os.path.exists(base_folder):
                print(f"⚠️ Folder not found: {base_folder}")
                continue
                
            if base_folder.endswith('real'):
                # Real数据: /code/real/multiview_subgraphs/1bathroom/uuid/
                pattern = os.path.join(base_folder, "multiview_subgraphs", "*", "*")
                scenes = glob.glob(pattern)
            else:
                # Sim数据: /code/sim_kitchen/FloorPlan1/2/
                scenes = []
                for root, dirs, files in os.walk(base_folder):
                    # 检查是否包含JSON和rgb文件夹
                    has_json = any(f.endswith('.json') for f in files)
                    has_rgb = 'rgb' in dirs
                    if has_json and has_rgb:
                        scenes.append(root)
            
            all_scenes.extend(scenes)
            print(f"📁 Found {len(scenes)} scenes in {base_folder}")
        
        print(f"🎯 Total scenes found: {len(all_scenes)}")
        return all_scenes
    
    def load_scene_data(self, scene_folder):
        """加载单个场景的JSON和图像"""
        # 找JSON文件
        json_files = glob.glob(os.path.join(scene_folder, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON file in {scene_folder}")
        
        json_file = json_files[0]  # 取第一个
        
        # 读取JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        
        # 找RGB图像
        rgb_folder = os.path.join(scene_folder, 'rgb')
        if not os.path.exists(rgb_folder):
            raise FileNotFoundError(f"No rgb folder in {scene_folder}")
        
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(rgb_folder, "*.jpg")) + \
                     glob.glob(os.path.join(rgb_folder, "*.png"))
        
        if not image_files:
            raise FileNotFoundError(f"No images in {rgb_folder}")
        
        return scene_data, sorted(image_files)
    
    def create_label_mapping(self, nodes):
        """创建label到编号名称的映射"""
        label_counts = defaultdict(int)
        label_mapping = {}
        
        for node in nodes:
            label = node.get("label", "unknown")
            label_counts[label] += 1
        
        # 为每个node分配编号名称
        current_counts = defaultdict(int)
        for node in nodes:
            label = node.get("label", "unknown")
            node_id = node.get("id", "")
            
            if label_counts[label] > 1:
                current_counts[label] += 1
                mapped_name = f"{label}{current_counts[label]}"
            else:
                mapped_name = label
            
            label_mapping[node_id] = mapped_name
        
        return label_mapping
    
    def simplify_graph_structure(self, scene_data):
        """将复杂的图结构转换为简化格式"""
        # 获取基本信息
        task_instruction = scene_data.get("task_instruction", "")
        action_type = scene_data.get("action_type", "")
        function_type = scene_data.get("function_type", "")
        
        nodes = scene_data.get("nodes", [])
        edges = scene_data.get("edges", [])
        
        # 创建label映射
        label_mapping = self.create_label_mapping(nodes)
        
        # 简化nodes
        simplified_nodes = list(label_mapping.values())
        
        # 简化edges
        simplified_edges = []
        for edge in edges:
            obj1_id = edge.get("object1", {}).get("id", "")
            obj2_id = edge.get("object2", {}).get("id", "")
            
            obj1_name = label_mapping.get(obj1_id, "unknown")
            obj2_name = label_mapping.get(obj2_id, "unknown")
            
            # 标准化functional relationship
            func_rel = edge.get("functional_relationship", "")
            standardized_func_rel = self.func_rel_mapping.get(func_rel, func_rel)
            
            simplified_edge = {
                "functional_relationship": standardized_func_rel,
                "object1": obj1_name,
                "object2": obj2_name,
                "spatial_relations": edge.get("spatial_relations", []),
                "is_touching": edge.get("is_touching", False)
            }
            simplified_edges.append(simplified_edge)
        
        # 构建简化的图结构
        simplified_graph = {
            "task_instruction": task_instruction,
            "nodes": simplified_nodes,
            "edges": simplified_edges,
            "action_type": action_type,
            "function_type": function_type
        }
        
        return simplified_graph
    
    def concatenate_images(self, image_files, max_width=2048):
        """拼接多个视角图像为单张图像 (复用benchmark代码)"""
        if not image_files:
            return None
        
        # 读取图片
        images = []
        for img_path in image_files:
            img = Image.open(img_path)
            images.append(img)
        
        n_images = len(images)
        
        # 计算网格布局
        if n_images <= 2:
            cols, rows = n_images, 1
        elif n_images <= 4:
            cols, rows = 2, 2
        elif n_images <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3
        
        # 计算平均图片宽高比
        avg_ratio = sum(img.width / img.height for img in images) / len(images)
        
        # 根据图片比例调整cell尺寸
        if avg_ratio > 1.3:  # 横向图片
            base_width = min(max_width // cols, 700)
            cell_width = base_width
            cell_height = int(base_width / avg_ratio) + 20
        else:  # 方形或竖向图片
            base_height = 400
            cell_height = base_height + 20
            cell_width = int(base_height * avg_ratio)
        
        # 图片resize
        resized_images = []
        for img in images:
            available_width = cell_width - 10
            available_height = cell_height - 35
            ratio = min(available_width / img.width, available_height / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        # 创建网格
        grid_width = cols * cell_width
        grid_height = rows * cell_height
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # 拼接图片
        for i, img in enumerate(resized_images):
            if i >= n_images:
                break
                
            row = i // cols
            col = i % cols
            
            cell_x = col * cell_width
            cell_y = row * cell_height
            
            img_x = cell_x + (cell_width - img.width) // 2
            img_y = cell_y + 3
            
            grid_img.paste(img, (img_x, img_y))
        
        return grid_img
    
    def process_single_scene(self, scene_folder):
        """处理单个场景，返回数据集条目"""
        try:
            # 加载数据
            scene_data, image_files = self.load_scene_data(scene_folder)
            
            # 如果是列表，取第一个元素
            if isinstance(scene_data, list):
                scene_data = scene_data[0]
            
            # 简化图结构
            simplified_graph = self.simplify_graph_structure(scene_data)
            
            # 拼接图像
            concat_image = self.concatenate_images(image_files)
            if concat_image is None:
                print(f"❌ Failed to concatenate images for {scene_folder}")
                return None
            
            # 构建数据集条目
            entry = {
                "images": [concat_image],
                "problem": f"<image>Task instruction: {simplified_graph['task_instruction']}",
                "answer": json.dumps(simplified_graph, ensure_ascii=False)
            }
            
            return entry
            
        except Exception as e:
            print(f"❌ Error processing {scene_folder}: {e}")
            return None
    
    def process_all_scenes(self, base_folders, max_samples=None):
        """处理所有场景"""
        # 查找所有场景
        all_scenes = self.find_all_scenes(base_folders)
        
        if max_samples:
            all_scenes = all_scenes[:max_samples]
            print(f"🔬 Processing first {max_samples} scenes for testing")
        
        # 处理每个场景
        processed_data = []
        
        for scene_folder in tqdm(all_scenes, desc="Processing scenes"):
            entry = self.process_single_scene(scene_folder)
            if entry:
                processed_data.append(entry)
        
        print(f"✅ Successfully processed {len(processed_data)} scenes")
        return processed_data
    
    def create_dataset(self, processed_data, train_ratio=0.95):
        """创建训练和验证数据集"""
        # 随机打乱
        random.shuffle(processed_data)
        
        # 分割
        total_size = len(processed_data)
        train_size = int(total_size * train_ratio)
        
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:]
        
        print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        # 创建Dataset对象
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # 创建DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        return dataset_dict

# 测试代码
if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    
    # 定义数据路径
    base_folders = [
        "/code/real",
        "/code/sim_kitchen", 
        "/code/sim_bathroom",
        "/code/sim_bedroom",
        "/code/sim_livingroom"
    ]
    
    # 创建处理器
    processor = SceneGraphProcessor()
    
    print("🚀 Scene Graph Dataset Processor")
    print("=" * 50)
    
    # 先测试10个样本
    print("🔬 Testing with 10 samples...")
    processed_data = processor.process_all_scenes(base_folders, max_samples=10)
    
    if processed_data:
        # 创建数据集
        dataset = processor.create_dataset(processed_data)
        
        # 打印样本
        print("\n📋 Sample data:")
        print(f"Problem: {dataset['train'][0]['problem']}")
        print(f"Answer: {dataset['train'][0]['answer'][:200]}...")
        
        # 保存测试数据集
        dataset.save_to_disk("./test_scene_graph_dataset")
        print("💾 Test dataset saved to ./test_scene_graph_dataset")
        
        print("✅ Test completed! Please verify the data before processing all scenes.")
    else:
        print("❌ No data processed successfully!")
