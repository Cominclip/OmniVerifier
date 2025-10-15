import os
import json
import argparse
import re
import openai
import time
import random

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


base_url = ""
api_version = "2024-03-01-preview"
ak = ""
model_name = "gpt-4.1-2025-04-14"
max_tokens = 4000  

client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=ak,
)

ANSWER_JUDGE_PROMPT = '''You are an expert evaluator. Given a reference answer to a question and a student's response, your task is to determine whether the student's response aligns with the reference answer. If it does, respond with 'Yes'; otherwise, respond with 'No'. Please note that your response must be either 'Yes' or 'No'—no additional text is allowed.
<Question>
{}
<Student's Response>
{}
<Reference Answer>
{}
'''

def extract_detailed_judgement(text):
    if isinstance(text, list):
        if '</think>' in text[0]:
            text = text[0].split('</think>')[1]
        else:
            text = text[0]
    try:
        if text.startswith("```json"):
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL)
        data = json.loads(text)
        return data.get("explanation", None)
    except json.JSONDecodeError:
        match = re.search(r'"explanation"\s*:\s*"([^"]*?)"', text)
        if match:
            return match.group(1)
    return None

def answer_judge(query, answer, ref_answer):
    max_retry_num = 5
    prompt = ANSWER_JUDGE_PROMPT.format(query, answer, ref_answer)

    for _ in range(max_retry_num):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=0.0,
                timeout=120,
            )
            matched_results_raw = completion.choices[0].message.content
            matched_results_raw = matched_results_raw.strip().lower()
            assert matched_results_raw in ["yes", "no"], f"matched_results_raw {matched_results_raw}"
            if matched_results_raw == "yes":
                score = 1
            else:
                score = 0
            return score
        except Exception:
            sleep_time = random.random() * 90 + 30
            time.sleep(sleep_time)
            continue


model_based_results = []
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
                if extracted_value:
                    num_correct += 1
                else: 
                    detailed_judgement = extract_detailed_judgement(item['output'])
                    num_correct += answer_judge(item['question'].split('Provide your evaluation')[0], detailed_judgement, item['explanation'])
                    
    model_based_results.append({
        "task": task,
        "num_correct": num_correct,
        "total": len(task_data),
        "accuracy": num_correct / len(task_data) if len(task_data) > 0 else 0
    })

print(sum(x['accuracy'] for x in model_based_results) / len(model_based_results))


with open(args.model_response.replace('.json', '_model_based.json'), 'w', encoding='utf-8') as f:
    json.dump(model_based_results, f, ensure_ascii=False, indent=4)
