#!/usr/bin/env python3
import pandas as pd
from datasets import load_dataset

print("🔍 Checking dataset format...")

# Try to load the dataset
try:
    dataset = load_dataset("/code/scenes/data")
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Dataset info:")
    print(f"  - Train samples: {len(dataset['train']) if 'train' in dataset else 'N/A'}")
    print(f"  - Validation samples: {len(dataset['validation']) if 'validation' in dataset else 'N/A'}")
    
    if 'train' in dataset and len(dataset['train']) > 0:
        print(f"📋 First training sample columns:")
        sample = dataset['train'][0]
        for key, value in sample.items():
            if key == 'images':
                print(f"  - {key}: {type(value)} (image)")
            else:
                print(f"  - {key}: {str(value)[:100]}...")
                
    print(f"🎯 Expected columns: images, problem, answer")
                
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    
    # Try loading individual parquet files
    print("\n🔍 Checking individual parquet files...")
    try:
        train_df = pd.read_parquet("/code/scenes/data/train-00000-of-00003.parquet")
        print(f"✅ Train parquet loaded: {len(train_df)} rows")
        print(f"📋 Columns: {list(train_df.columns)}")
        print(f"📄 First row sample:")
        for col in train_df.columns:
            print(f"  - {col}: {str(train_df[col].iloc[0])[:100]}...")
    except Exception as e2:
        print(f"❌ Error loading parquet: {e2}")
