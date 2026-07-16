import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up dark style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

# Hide axes
ax.axis('off')
ax.set_xlim(0, 24)
ax.set_ylim(0, 13)

# Title
ax.text(12, 12, 'Mecanismo de Atenção: Standard vs. FlashAttention', 
        color='#EEEEEE', fontsize=15, fontweight='bold', ha='center')

# Left side: Standard Attention (Memory-Bound)
ax.text(6, 11, 'ATENÇÃO TRADICIONAL (Standard Attention)', color='#FF2E63', fontsize=12, fontweight='bold', ha='center')

# Draw HBM and SRAM boxes on left
# HBM Left
hbm_left = patches.FancyBboxPatch((1.5, 2.5), 3, 7, boxstyle="round,pad=0.2", edgecolor='#393E46', facecolor='#222831', linewidth=1.5)
ax.add_patch(hbm_left)
ax.text(3, 9.2, 'GPU HBM\n(Memória Global\nLenta)', color='#EEEEEE', fontsize=10, fontweight='bold', ha='center')

# SRAM Left
sram_left = patches.FancyBboxPatch((7.5, 4.5), 2.5, 4, boxstyle="round,pad=0.2", edgecolor='#393E46', facecolor='#303841', linewidth=1.5)
ax.add_patch(sram_left)
ax.text(8.75, 8.0, 'GPU SRAM\n(Memória local\nRápida)', color='#EEEEEE', fontsize=10, fontweight='bold', ha='center')

# Flows for Left (Standard)
# 1. Read Q, K (HBM to SRAM)
ax.annotate('', xy=(7.3, 7), xytext=(4.7, 8), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=2))
ax.text(6.0, 7.8, '1. Lê Q, K', color='#00ADB5', fontsize=8, ha='center', rotation=-22)

# 2. Write N x N matrix (SRAM to HBM)
ax.annotate('', xy=(4.7, 5.5), xytext=(7.3, 6.5), arrowprops=dict(arrowstyle="->", color='#FF2E63', lw=2))
ax.text(6.0, 5.6, '2. Grava Matriz S\nO(N²) em HBM', color='#FF2E63', fontsize=8, ha='center', rotation=-22)

# 3. Read S, V (HBM to SRAM)
ax.annotate('', xy=(7.3, 5), xytext=(4.7, 4.0), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=2))
ax.text(6.0, 4.2, '3. Lê S, V', color='#00ADB5', fontsize=8, ha='center', rotation=22)

# 4. Write Output (SRAM to HBM)
ax.annotate('', xy=(4.7, 3.0), xytext=(7.3, 3.5), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=2))
ax.text(6.0, 3.0, '4. Grava Output O', color='#00ADB5', fontsize=8, ha='center', rotation=11)


# Right side: FlashAttention (Compute-Bound, SRAM Tiling)
ax.text(18, 11, 'FLASHATTENTION (Tiling & Online Softmax)', color='#00ADB5', fontsize=12, fontweight='bold', ha='center')

# HBM Right
hbm_right = patches.FancyBboxPatch((13.5, 2.5), 3, 7, boxstyle="round,pad=0.2", edgecolor='#393E46', facecolor='#222831', linewidth=1.5)
ax.add_patch(hbm_right)
ax.text(15, 9.2, 'GPU HBM\n(Memória Global\nLenta)', color='#EEEEEE', fontsize=10, fontweight='bold', ha='center')

# SRAM Right
sram_right = patches.FancyBboxPatch((19.5, 3.5), 3, 5, boxstyle="round,pad=0.2", edgecolor='#00ADB5', facecolor='#1A3738', linewidth=2)
ax.add_patch(sram_right)
ax.text(21, 8.0, 'GPU SRAM\n(Tiling em blocos)', color='#00ADB5', fontsize=10, fontweight='bold', ha='center')

# Flows for Right (FlashAttention)
# 1. Load blocks (HBM to SRAM)
ax.annotate('', xy=(19.3, 6.8), xytext=(16.7, 7.3), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=2))
ax.text(18.0, 7.3, '1. Lê blocos de\nQ, K, V', color='#00ADB5', fontsize=8, ha='center', rotation=-11)

# 2. Computa incrementalmente no SRAM (Online Softmax)
ax.text(21, 5.5, 'Computação\nIncremental\nSem salvar O(N²)\nem HBM', color='#EEEEEE', fontsize=8, ha='center', style='italic')

# 3. Write Output directly (SRAM to HBM)
ax.annotate('', xy=(16.7, 4.3), xytext=(19.3, 4.8), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=2))
ax.text(18.0, 4.1, '2. Grava Output O\ndiretamente', color='#00ADB5', fontsize=8, ha='center', rotation=-11)


# Bottom explanation card
info_box = patches.FancyBboxPatch((2, 0.5), 20, 1.4, boxstyle="round,pad=0.2", edgecolor='#393E46', facecolor='#1E1E1E', linewidth=1)
ax.add_patch(info_box)
ax.text(12, 1.2, 'Diferença Fundamental: A atenção tradicional sofre com afunilamento de leitura/escrita na HBM devido à matriz gigante O(N²).', 
        color='#EEEEEE', fontsize=8.5, fontweight='bold', ha='center')
ax.text(12, 0.7, 'O FlashAttention divide as matrizes Q, K, V em blocos (tiling), resolve o Softmax incrementalmente no SRAM e reduz acessos lentos de I/O.', 
        color='#B2B2B2', fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('/home/vini/ai_course/images/flash_attention_mechanism.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print("FlashAttention diagram generated successfully.")
