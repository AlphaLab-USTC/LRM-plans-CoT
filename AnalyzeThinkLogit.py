#%%
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model/DeepSeek-R1-Distill-Qwen-1.5B")

# %%
think_token_logits, think_token_probs, think_token_logits_mean, think_token_probs_mean = {}, {}, {}, {}
eos_token_logits, eos_token_probs, eos_token_logits_mean, eos_token_probs_mean = {}, {}, {}, {}
random_token_logits, random_token_probs, random_token_logits_mean, random_token_probs_mean = {}, {}, {}, {}
all_token_logits_mean, all_token_probs_mean = {}, {}

# for strength in (-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2):
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
    logits_probs_file_path = f"Assets/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/steering_by_strength/MATH500/token_logits_probs_{strength}.npy"
    logits_probs = np.load(logits_probs_file_path, allow_pickle=True).item()
    print(logits_probs)
    think_token_logits[strength] = logits_probs['think_token_logits']
    think_token_probs[strength] = logits_probs['think_token_probs']
    eos_token_logits[strength] = logits_probs['eos_token_logits']
    eos_token_probs[strength] = logits_probs['eos_token_probs']
    random_token_logits[strength] = logits_probs['random_token_logits']
    random_token_probs[strength] = logits_probs['random_token_probs']
    all_token_logits_mean[strength] = logits_probs['mean_last_token_logits']
    all_token_probs_mean[strength] = logits_probs['mean_last_token_probs']
    
    think_token_logits_mean[strength] = np.mean(logits_probs['think_token_logits'], axis=0)
    think_token_probs_mean[strength] = np.mean(logits_probs['think_token_probs'], axis=0)
    eos_token_logits_mean[strength] = np.mean(logits_probs['eos_token_logits'], axis=0)
    eos_token_probs_mean[strength] = np.mean(logits_probs['eos_token_probs'], axis=0)
    random_token_logits_mean[strength] = np.mean(logits_probs['random_token_logits'], axis=0)
    random_token_probs_mean[strength] = np.mean(logits_probs['random_token_probs'], axis=0)
    # break
#%%
add_strength = 0.2
minus_strength = -0.2
topk = 20

add_probs = all_token_probs_mean[add_strength] - all_token_probs_mean[0.0]
minus_probs = all_token_probs_mean[minus_strength] - all_token_probs_mean[0.0]
print("add_probs: ", add_probs)
print("minus_probs: ", minus_probs)

add_sorted_indices_decreasing = np.argsort(add_probs)[::-1]
minus_sorted_indices_decreasing = np.argsort(minus_probs)[::-1]
add_sorted_indices_increasing = np.argsort(add_probs)
minus_sorted_indices_increasing = np.argsort(minus_probs)
print("add_sorted_indices_decreasing: ", add_sorted_indices_decreasing)
print("minus_sorted_indices_decreasing: ", minus_sorted_indices_decreasing)
    
def decode_topk_tokens(sorted_indices, token_probs, k=10):
    """
    解码topk个token
    """
    for idx, indice in enumerate(sorted_indices[:k]):
        print("Probability:", token_probs[indice], "Token", tokenizer.decode([indice]))

print(f"\nadd_strength ({add_strength}) (TopK tokens): ")
decode_topk_tokens(add_sorted_indices_decreasing, add_probs, topk)
print(f"\nadd_strength ({add_strength}) (Lowest K tokens): ")
decode_topk_tokens(add_sorted_indices_increasing, add_probs, topk)


print(f"\nminus_strength ({minus_strength}) (TopK tokens): ")
decode_topk_tokens(minus_sorted_indices_decreasing, minus_probs, topk)
print(f"\nminus_strength ({minus_strength}) (Lowest K tokens): ")
decode_topk_tokens(minus_sorted_indices_increasing, minus_probs, topk)

#%%
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
    print(strength)
    print("think_token_logits_mean: ", think_token_logits_mean[strength])
    print("think_token_probs_mean: ", think_token_probs_mean[strength])
    print("eos_token_logits_mean: ", eos_token_logits_mean[strength])
    print("eos_token_probs_mean: ", eos_token_probs_mean[strength])
    print("random_token_logits_mean: ", random_token_logits_mean[strength])
    print("random_token_probs_mean: ", random_token_probs_mean[strength])
#%%
strengths = [-0.2, -0.1, 0.0, 0.1, 0.2]
plt.figure(figsize=(12, 8))

# Plot the logits and probs of the think token.
plt.subplot(2, 1, 1)
plt.plot(strengths, [think_token_logits_mean[s] for s in strengths], 'o-', label='think token logits')
# plt.plot(strengths, [eos_token_logits_mean[s] for s in strengths], 'o-', label='eos token logits')
plt.plot(strengths, [random_token_logits_mean[s] for s in strengths], 'o-', label='random token logits')
plt.xlabel('Steering Strength')
plt.ylabel('Logits')
plt.title('Token Logits vs Steering Strength')
plt.legend()
plt.grid(True)

# Draw the probs of the think token.
plt.subplot(2, 1, 2)
plt.plot(strengths, [np.log(think_token_probs_mean[s] * 5e15) for s in strengths], 'o-', label='think token probs')
# plt.plot(strengths, [np.log(eos_token_probs_mean[s] * 5e15) for s in strengths], 'o-', label='eos token probs')
plt.plot(strengths, [np.log(random_token_probs_mean[s] * 5e15) for s in strengths], 'o-', label='random token probs')
plt.xlabel('Steering Strength')
plt.ylabel('Probability')
plt.title('Token Probabilities vs Steering Strength')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#%%
print(logits_probs.keys())
# %%
def plot_token_distribution(token_logits_or_probs, name='None', strengths=None, use_log=False, scale=1, save_path=None):
    """
    Plot the distribution of tokens.
    
    Parameters:
    token_logits_or_probs (dict): A dictionary containing the token logits or probabilities for different strengths
    name (str): The name of the data, used for the title and labels
    strengths (list): The list of strength values to plot, default is [-0.2, -0.1, 0.0, 0.1, 0.2]
    use_log (bool): Whether to take the logarithm of the logits, default is False
    save_path (str): The path to save the image, if None, the image will be displayed instead of saved
    """
    import matplotlib.pyplot as plt
    
    # If strengths are not specified, use the default value.
    if strengths is None:
        strengths = [-0.2, -0.1, 0.0, 0.1, 0.2]
    
    plt.figure(figsize=(10, 6))
    
    for strength in strengths:
        data = token_logits_or_probs[strength] * scale
        if use_log:
            data = np.log(data)
        plt.hist(data, bins=100, alpha=0.5, label=f'strength={strength}')
    
    plt.legend()
    plt.xlabel(f'{name} (log)' if use_log else f'{name}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {f"{name} (log)" if use_log else f"{name}"} '
              f'for Different Steering Strengths')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Usage example:
# plot_think_token_distribution(think_token_logits)  # Use default parameters
# plot_think_token_distribution(think_token_logits, strengths=[-0.2, 0.0, 0.2])  # Specify strength values
# plot_think_token_distribution(think_token_logits, use_log=True)  # Use logarithmic scale
# plot_think_token_distribution(think_token_logits, save_path='distribution.png')  # Save the image
#%%
from matplotlib import font_manager
font_path = 'Assets/Times New Roman Bold.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

def plot_token_distribution(token_logits_or_probs, name='None', strengths=None, 
                        figsize=(12, 8), use_log=False, 
                          scale=1, save_path=None):
    """
    Plot the distribution of tokens, with beautiful fonts and styles.
    
    Parameters:
    token_logits_or_probs (dict): A dictionary containing the token logits or probabilities for different strengths
    name (str): The name of the data, used for the title and labels
    strengths (list): The list of strength values to plot, default is [-0.2, -0.1, 0.0, 0.1, 0.2]
    font_prop: The font property, default is None
    figsize (tuple): The size of the chart, default is (12, 8)
    use_log (bool): Whether to take the logarithm of the data, default is False
    scale (float): The scaling factor of the data, default is 1
    save_path (str): The path to save the image, if None, the image will be displayed instead of saved
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    # If strengths are not specified, use the default value.
    if strengths is None:
        strengths = [-0.2, -0.1, 0.0, 0.1, 0.2]
    
    # Create a custom color mapping.
    colors = ['#c4d7ef', '#a7d2cb', '#3a76a3', '#ff8248', '#eab299'][::-1]
    
    try:
        plt.figure(figsize=figsize)
        
        # Configure the axis borders and bold all borders.
        ax = plt.gca()
        ax.spines['left'].set_linewidth(3.8)
        ax.spines['bottom'].set_linewidth(3.8)
        ax.spines['right'].set_linewidth(3.8)
        ax.spines['top'].set_linewidth(3.8)
        
        # Draw histograms of different intensities.
        for i, strength in enumerate(strengths):
            if strength in token_logits_or_probs:
                data = token_logits_or_probs[strength] * scale
                if use_log:
                    data = np.log(data)
                
                color_idx = i % len(colors)
                plt.hist(data, bins=100, alpha=0.7,
                        color=colors[color_idx], 
                        label=f'strength={strength}')
        
        # Set labels and titles, using custom fonts and sizes.
        plt.xlabel(f'{name} (log)' if use_log else f'{name}', 
                  fontproperties=font_prop, fontsize=32)
        plt.ylabel('Frequency', fontproperties=font_prop, fontsize=32)
        
        # Set the font and size of the scale labels.
        plt.xticks(fontproperties=font_prop, fontsize=28)
        plt.yticks(fontproperties=font_prop, fontsize=28)
        
        # Add grid lines.
        plt.grid(True, linestyle='--', alpha=0.4, linewidth=2)
        
        # Add a legend, using custom fonts and sizes.
        font_prop_large = font_manager.FontProperties(fname=font_path, size=32)
        plt.legend(prop=font_prop_large, fontsize=32, frameon=True, 
                  facecolor='white', edgecolor='#515b83', 
                  framealpha=0.9)
        
        plt.tight_layout()
        
        # Save or display the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'The chart has been saved to {save_path}')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"An error occurred during the plotting process: {e}")

#%%
plot_token_distribution(think_token_logits, name='think_token_logits', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=False)
plot_token_distribution(eos_token_logits, name='eos_token_logits', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=False)
plot_token_distribution(random_token_logits, name='random_token_logits', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=False)
# Calculate the average value of think_token_logits based on each strength.

#%%
plot_token_distribution(think_token_probs, name='think_token_probs', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=True, scale=1e20)
plot_token_distribution(eos_token_probs, name='eos_token_probs', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=True, scale=1e20)
plot_token_distribution(random_token_probs, name='random_token_probs', strengths=[-0.2, -0.1, 0.0, 0.1, 0.2], use_log=True, scale=1e20)

#%%



# %%
# Draw a frequency histogram for each strength.
# for strength in (-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2):
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
# for strength in (-0.2, -0.15):
# for strength in [-0.2]:
    # plt.hist(np.log(think_token_logits[strength]), bins=100, alpha=0.5, label=f'strength={strength}')
    plt.hist(think_token_logits[strength], bins=100, alpha=0.5, label=f'strength={strength}')
plt.legend()
plt.xlabel('log(think_token_logits)')
plt.ylabel('Frequency')
# plt.title('Distribution of log(think_token_logits) for Different Steering Strengths')
plt.title('Distribution of think_token_logits for Different Steering Strengths')
plt.show()
# %%
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
    plt.hist(np.log(1e7*think_token_probs[strength]), bins=100, alpha=0.5, label=f'strength={strength}')
plt.legend()
plt.xlabel('log(1e7 * think_token_probs)')
plt.ylabel('Frequency')
plt.title('Distribution of log(1e7 * think_token_probs) for Different Steering Strengths')
plt.show()
# %%
# eos_token_logits
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
    plt.hist(eos_token_logits[strength], bins=100, alpha=0.5, label=f'strength={strength}')
plt.legend()
plt.xlabel('eos_token_logits')
plt.ylabel('Frequency')
plt.title('Distribution of eos_token_logits for Different Steering Strengths')
plt.show()
#%%
# eos_token_probs
for strength in (-0.2, -0.1, 0.0, 0.1, 0.2):
    plt.hist(np.log(1e7*eos_token_probs[strength]), bins=100, alpha=0.5, label=f'strength={strength}')
plt.legend()
plt.xlabel('log(1e7 * eos_token_probs)')
plt.ylabel('Frequency')
plt.title('Distribution of log(1e7 * eos_token_probs) for Different Steering Strengths')
# %%
