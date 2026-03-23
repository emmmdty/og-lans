#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS Ontology Graph Builder

Constructs the event taxonomy graph from dataset schema files.

This script builds a NetworkX graph representing the hierarchical structure
of event types and their argument roles, which is essential for the OG-CNS
(Ontology-Graph Driven Contrastive Negative Sampling) algorithm.

Graph Structure:
    ROOT
    ├── EventType1
    │   ├── EventType1::Role1
    │   └── EventType1::Role2
    ├── EventType2
    │   └── EventType2::Role1
    └── ...

Usage:
    # Build graph for DuEE-Fin dataset
    python scripts/build_graph.py --dataset_name DuEE-Fin

    # Custom paths
    python scripts/build_graph.py --schema_path ./schema.json --output_path ./graph.gml

Output:
    GML format graph file compatible with NetworkX and other graph libraries.

Authors:
    OG-LANS Research Team
"""

# scripts/build_graph.py
import sys
import os
import json
import argparse
import networkx as nx
import logging

# 将项目根目录加入路径
sys.path.append(os.getcwd())
from oglans.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Build Taxonomy Graph for OG-LANS")
    
    # 1. 数据集名称：默认为 DuEE-Fin (对应目录 data/raw/DuEE-Fin)
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="DuEE-Fin", 
        help="Name of the dataset directory (e.g., DuEE-Fin, ACE05). Default: DuEE-Fin"
    )
    
    # 2. Schema 路径：默认根据命名规则自动生成
    parser.add_argument(
        "--schema_path", 
        type=str, 
        default=None, 
        help="Path to schema JSON. If None, auto-constructed from dataset_name."
    )
    
    # 3. 输出路径：默认根据命名规则自动生成
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Path to save GML. If None, auto-constructed from dataset_name."
    )
    
    return parser.parse_args()

def build_graph(schema_path):
    """核心构建逻辑"""
    G = nx.Graph()
    G.add_node("ROOT", type="root")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        # 兼容单行 JSON 和多行 JSONL 格式
        try:
            # 尝试作为整个 JSON 读取
            schemas = json.load(f)
        except json.JSONDecodeError:
            # 回退到 JSONL (每行一个 JSON)
            f.seek(0)
            lines = f.readlines()
            schemas = [json.loads(line) for line in lines]

    # 遍历 Schema 构建图谱
    for schema in schemas:
        etype = schema['event_type']
        G.add_node(etype, type="event_type")
        G.add_edge("ROOT", etype)

        for role_obj in schema['role_list']:
            role_name = role_obj['role']
            # 使用 "事件::角色" 唯一ID防止跨事件混淆
            node_id = f"{etype}::{role_name}"
            G.add_node(node_id, type="role", role_name=role_name)
            G.add_edge(etype, node_id)
            
    return G

def main():
    args = parse_args()
    
    # === 路径自动构建逻辑 ===
    dataset_name = args.dataset_name
    
    # [关键修改] 生成文件名专用的基础名称：全小写 + 将连字符替换为下划线
    # 例如: "DuEE-Fin" -> "duee_fin"
    filename_base = dataset_name.lower().replace("-", "_")
    
    project_root = os.getcwd()

    # 1. 确定 Schema 路径
    if args.schema_path:
        schema_path = args.schema_path
    else:
        # 目录保持原样 (data/raw/DuEE-Fin)，文件名变更为下划线格式
        # 结果: data/raw/DuEE-Fin/duee_fin_event_schema.json
        schema_filename = f"{filename_base}_event_schema.json"
        schema_path = os.path.join(project_root, "data", "raw", dataset_name, schema_filename)

    # 2. 确定 输出 GML 路径
    if args.output_path:
        output_path = args.output_path
    else:
        # 结果: data/schemas/duee_fin_graph.gml
        output_dir = os.path.join(project_root, "data", "schemas")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{filename_base}_graph.gml"
        output_path = os.path.join(output_dir, output_filename)

    # === 初始化日志 ===
    log_dir = os.path.join("logs", "graph_build", filename_base)
    logger = setup_logger(f"GraphBuilder-{dataset_name}", log_dir)
    
    logger.info(f"🚀 Starting graph build for dataset: {dataset_name}")
    logger.info(f"📂 Schema Input: {schema_path}")
    logger.info(f"💾 Graph Output: {output_path}")

    # === 执行构建 ===
    try:
        G = build_graph(schema_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        nx.write_gml(G, output_path)
        logger.info(f"✅ Success! Graph saved to: {output_path}")
        logger.info(f"📊 Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    except Exception as e:
        logger.error(f"❌ Failed to build graph: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()