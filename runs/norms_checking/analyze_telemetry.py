import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Parse and flatten the JSONL log file
log_file_path = "runs/norms_checking/events.jsonl"  # Replace with your actual filename
extracted_rows = []

print("Extracting training logs...")
with open(log_file_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            # Only process lines containing the rich update metrics
            if item.get("type") == "update" and "payload" in item:
                row = {"step": item["step"]}
                # Flatten the internal payload keys
                for key, value in item["payload"].items():
                    row[key] = value
                extracted_rows.append(row)
        except json.JSONDecodeError:
            continue

# Convert to a DataFrame
df = pd.DataFrame(extracted_rows)
print(f"Successfully extracted {len(df)} update steps.")

# Save to CSV for easy inspection in Excel/Sheets
df.to_csv("extracted_telemetry.csv", index=False)
print("Saved clean data to 'extracted_telemetry.csv'")

# Use a clean plotting style
sns.set_theme(style="whitegrid")

# ==========================================
# PLOT 1: Gradient Norms (Actor vs Q-Network)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
if 'actor_grad_norm' in df.columns:
    ax.plot(df['step'], df['actor_grad_norm'], marker='o', color='royalblue', linewidth=2, label='Actor Gradient Norm')
if 'q_grad_norm' in df.columns:
    ax.plot(df['step'], df['q_grad_norm'], marker='s', color='crimson', linewidth=2, label='Q-Network Gradient Norm')

ax.set_title('Gradient Norms Evolution Over Training Steps', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Total Gradient Norm', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('gradient_norms.png', dpi=300)
plt.close()

# ==========================================
# PLOT 2: Parameter Norms (Weights/Biases Stability)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
if 'actor_param_norm' in df.columns:
    ax.plot(df['step'], df['actor_param_norm'], marker='o', color='purple', linewidth=2, label='Actor Params Norm')
if 'q_param_norm' in df.columns:
    ax.plot(df['step'], df['q_param_norm'], marker='s', color='teal', linewidth=2, label='Q-Network Params Norm')

ax.set_title('Parameter Norms Stability Over Time', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Parameter Norm Value', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('parameter_norms.png', dpi=300)
plt.close()

# ==========================================
# PLOT 3: Feature Activation Norms Across Layers
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
# Track how activation magnitudes look across consecutive layers
actor_layers = ['feat_actor_fc1', 'feat_actor_fc2']
q_layers = ['feat_q1_fc1', 'feat_q1_fc2', 'feat_q1_fc3']

for layer in actor_layers:
    if layer in df.columns:
        ax.plot(df['step'], df[layer], marker='o', linestyle='-', label=f'Actor: {layer.split("_")[-1].upper()}')

for layer in q_layers:
    if layer in df.columns:
        ax.plot(df['step'], df[layer], marker='s', linestyle='--', label=f'Q1: {layer.split("_")[-1].upper()}')

ax.set_title('Layer Feature Activation Norms (Representation Magnitude)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Activation Norm Value', fontsize=12)
ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('feature_norms.png', dpi=300)
plt.close()

print("Analysis complete! Plots generated: 'gradient_norms.png', 'parameter_norms.png', 'feature_norms.png'")