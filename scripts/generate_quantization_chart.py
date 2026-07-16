import matplotlib.pyplot as plt
import numpy as np

# Set style for a modern dark/professional look
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

# Data
quant_levels = ['FP32\n(Single Prec.)', 'FP16\n(Half Prec.)', 'INT8\n(8-bit Quant)', 'INT4\n(4-bit Quant)']
# Memory size for a 70B parameter model in GB
memory_sizes = [280, 140, 70, 35]
# Hypothetical Perplexity (lower is better, relatively small increase)
perplexity = [5.10, 5.11, 5.15, 5.35]

# Plot Memory Size (Bar chart)
color_size = '#00ADB5' # Teal
bars = ax1.bar(quant_levels, memory_sizes, color=color_size, alpha=0.6, width=0.4, label='Tamanho do Modelo (GB)')
ax1.set_ylabel('Espaço em Memória / RAM (GB)', color=color_size, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_size)
ax1.set_title('Trade-off de Quantização em LLMs (Ex: Modelo de 70B)', fontsize=14, fontweight='bold', pad=20, color='#EEEEEE')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height} GB',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', color=color_size, fontweight='bold')

# Create a second y-axis for Perplexity/Quality loss
ax2 = ax1.twinx()
color_perp = '#FF2E63' # Neon Pink
line = ax2.plot(quant_levels, perplexity, color=color_perp, marker='o', linewidth=3, markersize=8, label='Perplexidade (Menor = Melhor)')
ax2.set_ylabel('Perplexidade (Qualidade do Modelo)', color=color_perp, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_perp)

# Add values to the line plot
for i, txt in enumerate(perplexity):
    ax2.annotate(f'{txt}', (quant_levels[i], perplexity[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color_perp, fontweight='bold')

# Customize grid and spines
ax1.grid(True, linestyle='--', alpha=0.2)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Background color customization
fig.patch.set_facecolor('#1A1A1A')
ax1.set_facecolor('#1A1A1A')

# Caption / Info box
textstr = '\n'.join((
    'FP32/FP16: Alta fidelidade, requer hardware profissional (GPUs com VRAM grande).',
    'INT8: Compressão padrão, quase nenhuma perda de qualidade.',
    'INT4: Extrema compressão, ideal para rodar localmente em CPU/RAM de computadores comuns.'
))
props = dict(boxstyle='round', facecolor='#252525', alpha=0.8, edgecolor='#393E46')
ax1.text(0.05, 0.15, textstr, transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, color='#EEEEEE')

plt.tight_layout()
plt.savefig('/home/vini/ai_course/images/llm_quantization_tradeoffs.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print("Chart generated successfully at /home/vini/ai_course/images/llm_quantization_tradeoffs.png")
