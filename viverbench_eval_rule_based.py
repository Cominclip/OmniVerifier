import os
import json
import argparse
import re

parser = argparse.ArgumentParser(description='Process parquet files and save as JSON.')
parser.add_argument('--model_response', type=str, default='', help='Path to model_response.json')
args = parser.parse_args()

tasks = [
    'Concept Existence: Object',
    'Concept Existence: Attribute',
    'Concept Existence: Abstract Patterns',
    'Object Relationship: Spatial Relationship',
    'Object Relationship: Non-Spatial Relationship',
    'World Dynamics: Static Physics',
    'World Dynamics: Dynamic Physics',
    'Image Annotation: Bounding Box',
    'Image Annotation: Pointing',
    'Image Annotation: Counting',
    'State Value Evaluation: Maze',
    'State Value Evaluation: FrozenLake',
    'State Value Evaluation: Robotics',
    'State Value Evaluation: GUI',
    'STEM: Charts',
    'STEM: LaTeX',
    ]

with open(args.model_response, 'r', encoding='utf-8') as f:
    model_response = json.load(f)

rule_based_results = []
for task in tasks:
    num_correct = 0
    task_data = []
    for item in model_response:
        if item['task'] == task:
            task_data.append(item)
    for item in task_data:
        if 'output' not in item or item['output'] is None:
            continue
        if isinstance(item['output'], list):
            item['output'] = item['output'][0] 
        if '</think>' in item['output']:
            match = re.search(r'"answer"\s*:\s*(true|false)', item['output'].split('</think>')[1])
        else:
            match = re.search(r'"answer"\s*:\s*(true|false)', item['output'])
        if match and item['output'] != None:
            extracted_value = match.group(1).lower() == "true"
            is_same = extracted_value == item['answer']
            if is_same:
                num_correct += 1
    rule_based_results.append({
        "task": task,
        "num_correct": num_correct,
        "total": len(task_data),
        "accuracy": num_correct / len(task_data) if len(task_data) > 0 else 0
    })

print(sum(x['accuracy'] for x in rule_based_results) / len(rule_based_results))


with open(args.model_response.replace('.json', '_rule_based.json'), 'w', encoding='utf-8') as f:
    json.dump(rule_based_results, f, ensure_ascii=False, indent=4)