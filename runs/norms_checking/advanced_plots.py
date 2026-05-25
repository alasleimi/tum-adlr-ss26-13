import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your extracted data
df = pd.read_csv("extracted_telemetry.csv")
sns.set_theme(style="whitegrid")

# Create a multi-page or multi-figure setup
plt.figure(figsize=(20, 15))

# ==========================================
# 1. THE CRITIC OVERESTIMATION DIAGNOSTIC
# ==========================================
# Checks if your Critics are predicting wildly unrealistic values compared to targets
plt.subplot(3, 2, 1)
plt.plot(df['step'], df['q1_mean_mean'], label='Q1 Val Mean', color='teal')
plt.plot(df['step'], df['q2_mean_mean'], label='Q2 Val Mean', color='darkcyan')
plt.plot(df['step'], df['target_q_mean_mean'], label='Target Q Mean', linestyle='--', color='crimson')
plt.title('Critic Value Alignment (Overestimation Check)', fontsize=12, fontweight='bold')
plt.xlabel('Steps')
plt.ylabel('Q-Value')
plt.legend()

# ==========================================
# 2. ACTOR OUTPUT HEAD SPLIT (Mean vs Exploration)
# ==========================================
# Diagnoses if your policy variance (logstd) is collapsing too fast
plt.subplot(3, 2, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()
line1 = ax1.plot(df['step'], df['feat_actor_fc_mean_mean'], color='indigo', label='Action Mean Magnitude')
line2 = ax2.plot(df['step'], df['feat_actor_fc_logstd_mean'], color='darkorange', linestyle='--', label='Action LogStd (Exploration)')
ax1.set_ylabel('Mean Head Magnitude', color='indigo')
ax2.set_ylabel('LogStd Head Magnitude', color='darkorange')
plt.title('Actor Output Heads: Exploitation vs Exploration', fontsize=12, fontweight='bold')
ax1.set_xlabel('Steps')
# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# ==========================================
# 3. LAYER-BY-LAYER GRADIENT FLOW (Q-Network)
# ==========================================
# Verifies if gradients are actually flowing back to early layers or vanishing
plt.subplot(3, 2, 3)
plt.plot(df['step'], df['grad_q1_fc1.weight_mean'], label='Q1 FC1 (Early Layer)', color='purple')
plt.plot(df['step'], df['grad_q1_fc2.weight_mean'], label='Q1 FC2 (Mid Layer)', color='orchid')
plt.plot(df['step'], df['grad_q1_fc3.weight_mean'], label='Q1 FC3 (Output Layer)', color='deeppink')
plt.yscale('log')  # Gradients span orders of magnitude
plt.title('Critic Gradient Flow (Layer-by-Layer Mean)', fontsize=12, fontweight='bold')
plt.xlabel('Steps')
plt.ylabel('Gradient Magnitude (Log Scale)')
plt.legend()

# ==========================================
# 4. WEIGHT UPDATE RATIO (The "Learning Rate" Healthcheck)
# ==========================================
# Rule of thumb: updates should hover around 1e-3 (0.001). 
# > 1e-2 means updates are too violent; < 1e-4 means network is stalled.
plt.subplot(3, 2, 4)
plt.plot(df['step'], df['actor_update_norm_ratio_mean'], label='Actor Update Ratio', color='royalblue')
plt.plot(df['step'], df['q_update_norm_ratio_mean'], label='Q-Net Update Ratio', color='orangered')
plt.axhline(y=1e-3, color='gray', linestyle=':', label='Healthy Target (1e-3)')
plt.yscale('log')
plt.title('Update-to-Weight Ratio (Optimization Step Size Quality)', fontsize=12, fontweight='bold')
plt.xlabel('Steps')
plt.ylabel('Ratio (Log Scale)')
plt.legend()

# ==========================================
# 5. TEMPERATURE (ALPHA) ENTROPY DYNAMICS
# ==========================================
# Tracks how adaptive entropy regularizes the policy
plt.subplot(3, 2, 5)
ax3 = plt.gca()
ax4 = ax3.twinx()
line3 = ax3.plot(df['step'], df['alpha_mean'], color='forestgreen', label='Alpha (Temperature)')
line4 = ax4.plot(df['step'], df['policy_entropy_estimate_mean'], color='darkgoldenrod', linestyle='-.', label='Policy Entropy')
ax3.set_ylabel('Alpha Scale', color='forestgreen')
ax4.set_ylabel('Entropy Value', color='darkgoldenrod')
plt.title('Entropy Temperature (Alpha) vs Policy Randomness', fontsize=12, fontweight='bold')
ax3.set_xlabel('Steps')
lines_entropy = line3 + line4
labels_entropy = [l.get_label() for l in lines_entropy]
ax3.legend(lines_entropy, labels_entropy, loc='upper right')

# ==========================================
# 6. GLOBAL LOSS LANDSCAPE
# ==========================================
plt.subplot(3, 2, 6)
plt.plot(df['step'], df['actor_loss_mean'], label='Actor Loss', color='navy')
plt.plot(df['step'], df['q_loss_mean'], label='Total Q Loss', color='firebrick')
plt.title('Global Loss Progress', fontsize=12, fontweight='bold')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('sac_deep_diagnostics.png', dpi=300)
print("Advanced telemetry dashboard saved as 'sac_deep_diagnostics.png'")