import json
import os
from collections import defaultdict

base_eval_path = 'Data/Eval'
model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B', 'DeepSeek-R1-Distill-Qwen-32B', 'QwQ-32B']
# dataset_list = ['MATH500', 'AIME2024', 'Minerva', 'AMC', 'OlympiadBench']
dataset_list = ['MATH500', 'AIME2024', 'OlympiadBench']

results = defaultdict(dict)

for dataset in dataset_list:
    for model_name in model_name_list:
        eval_path = f'{base_eval_path}/{dataset}/{model_name}'
        if not os.path.exists(eval_path):
            continue
            
        overall_trend_path = f'{eval_path}/overall_trend_results_correct_vote_num8.json'
        if not os.path.exists(overall_trend_path):
            overall_trend_path = f'{eval_path}/overall_trend_results.json'
            if not os.path.exists(overall_trend_path):
                continue
            
        with open(overall_trend_path, 'r') as f:
            data = json.load(f)
            
        # Find the best performance.
        best_performance = max(data, key=lambda x: x['total_accuracy'])
        # Find the performance when strength = 0.
        strength_zero = next((item for item in data if item['strength'] == 0.0), None)
        
        results[dataset][model_name] = {
            'best_performance': {
                'strength': best_performance['strength'],
                'accuracy': best_performance['total_accuracy'],
                'reasoning_tokens': best_performance['average_reasoning_token_num'],
                'answer_tokens': best_performance['average_answer_token_num']
            },
            'strength_zero': {
                'accuracy': strength_zero['total_accuracy'],
                'reasoning_tokens': strength_zero['average_reasoning_token_num'],
                'answer_tokens': strength_zero['average_answer_token_num']
            } if strength_zero else None
        }

# Set column width.
MODEL_WIDTH = 15
CELL_WIDTH = 40
OVERALL_WIDTH = 40

# Print the form.
print("\n表格输出：")
separator = "-" * (MODEL_WIDTH + CELL_WIDTH * 5 + OVERALL_WIDTH + 7)
print(separator)

# Print the header.
headers = ["Model"] + dataset_list + ["Overall"]
header_line = f"|{headers[0]:^{MODEL_WIDTH}}|"
for h in headers[1:-1]:
    header_line += f"{h:^{CELL_WIDTH}}|"
header_line += f"{headers[-1]:^{OVERALL_WIDTH}}|"
print(header_line)
print(separator)

def format_cell(zero_data, best_data):
    if zero_data:
        zero_str = f"{zero_data['accuracy']:.4f}(0.00+{zero_data['reasoning_tokens']:.0f})"
    else:
        zero_str = "/"
        
    best_str = f"{best_data['accuracy']:.4f}({best_data['strength']:.2f}+{best_data['reasoning_tokens']:.0f})"
    return f"{zero_str}/{best_str}"

# Calculate the average performance of each model.
for model_name in model_name_list:
    short_name = model_name.replace('DeepSeek-R1-Distill-', '')
    row = [f"|{short_name:^{MODEL_WIDTH}}|"]
    zero_accs = []
    best_accs = []
    zero_tokens = []
    best_tokens = []
    
    for dataset in dataset_list:
        if dataset in results and model_name in results[dataset]:
            data = results[dataset][model_name]
            cell = format_cell(data['strength_zero'], data['best_performance'])
            
            if data['strength_zero']:
                zero_accs.append(data['strength_zero']['accuracy'])
                best_accs.append(data['best_performance']['accuracy'])
                zero_tokens.append(data['strength_zero']['reasoning_tokens'])
                best_tokens.append(data['best_performance']['reasoning_tokens'])
        else:
            cell = "/"
            
        row.append(f"{cell:^{CELL_WIDTH}}|")
    
    # Calculate the average.
    if zero_accs and best_accs:
        avg_zero_acc = sum(zero_accs) / len(zero_accs)
        avg_best_acc = sum(best_accs) / len(best_accs)
        avg_zero_tokens = sum(zero_tokens) / len(zero_tokens)
        avg_best_tokens = sum(best_tokens) / len(best_tokens)
        overall = f"{avg_zero_acc:.4f}(0.00+{avg_zero_tokens:.0f})/{avg_best_acc:.4f}(0.05+{avg_best_tokens:.0f})"
    else:
        overall = "/"
        
    row.append(f"{overall:^{OVERALL_WIDTH}}|")
    print("".join(row))
print(separator) 