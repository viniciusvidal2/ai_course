import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up dark style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

ax.axis('off')
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)

# Title
ax.text(11, 11, 'Conceito e Arquitetura do LangGraph', color='#EEEEEE', fontsize=14, fontweight='bold', ha='center')

# State (Shared memory box) in center top
state_box = patches.FancyBboxPatch((7.5, 8.2), 7, 1.4, boxstyle="round,pad=0.1", edgecolor='#00ADB5', facecolor='#1E2E3A', linewidth=1.5)
ax.add_patch(state_box)
ax.text(11, 9.1, 'State (Estado Compartilhado)', color='#00ADB5', fontsize=10, fontweight='bold', ha='center')
ax.text(11, 8.6, 'Esquema de dados atualizado a cada nó (Ex: {"messages": [...]})', color='#B2B2B2', fontsize=7.5, ha='center')

# Draw Nodes
# Node: START
start_node = patches.Circle((3, 5), radius=0.8, edgecolor='#393E46', facecolor='#303841', linewidth=1.5)
ax.add_patch(start_node)
ax.text(3, 4.9, 'START', color='#EEEEEE', fontsize=9, fontweight='bold', ha='center')

# Node: Agent (LLM)
agent_node = patches.FancyBboxPatch((7.5, 4.2), 3, 1.6, boxstyle="round,pad=0.1", edgecolor='#FF2E63', facecolor='#351E28', linewidth=1.5)
ax.add_patch(agent_node)
ax.text(9, 5.2, 'Nó: Agent', color='#FF2E63', fontsize=9.5, fontweight='bold', ha='center')
ax.text(9, 4.6, 'Processa e decide', color='#B2B2B2', fontsize=7.5, ha='center')

# Node: Tool Exec
tool_node = patches.FancyBboxPatch((13.5, 4.2), 3, 1.6, boxstyle="round,pad=0.1", edgecolor='#EEEEEE', facecolor='#252A34', linewidth=1.5)
ax.add_patch(tool_node)
ax.text(15, 5.2, 'Nó: Action/Tool', color='#EEEEEE', fontsize=9.5, fontweight='bold', ha='center')
ax.text(15, 4.6, 'Executa ferramenta', color='#B2B2B2', fontsize=7.5, ha='center')

# Node: END
end_node = patches.Circle((20, 5), radius=0.8, edgecolor='#393E46', facecolor='#303841', linewidth=1.5)
ax.add_patch(end_node)
ax.text(20, 4.9, 'END', color='#EEEEEE', fontsize=9, fontweight='bold', ha='center')

# Connective Edges / Arrows
# Start -> Agent
ax.annotate('', xy=(7.3, 5.0), xytext=(3.9, 5.0), arrowprops=dict(arrowstyle="->", color='#EEEEEE', lw=1.5))
# Agent -> Tool (Conditional Edge)
ax.annotate('', xy=(13.3, 5.5), xytext=(10.7, 5.5), arrowprops=dict(arrowstyle="->", color='#FF2E63', lw=1.5))
ax.text(12, 5.7, 'Chama ferramenta', color='#FF2E63', fontsize=7.5, ha='center')

# Tool -> Agent (Loop back)
ax.annotate('', xy=(9.0, 4.1), xytext=(15.0, 4.1), arrowprops=dict(arrowstyle="->", color='#EEEEEE', lw=1.5, connectionstyle="arc3,rad=-0.5"))
ax.text(12, 2.7, 'Retorna resultado ao Estado', color='#B2B2B2', fontsize=7.5, ha='center')

# Agent -> End (Conditional Edge)
ax.annotate('', xy=(19.1, 5.2), xytext=(10.7, 5.2), arrowprops=dict(arrowstyle="->", color='#00ADB5', lw=1.5, connectionstyle="arc3,rad=0.3"))
ax.text(14.9, 6.7, 'Resposta final obtida', color='#00ADB5', fontsize=7.5, ha='center')

# Bottom explanation
info_box = patches.FancyBboxPatch((1.5, 0.6), 19, 1.2, boxstyle="round,pad=0.1", edgecolor='#393E46', facecolor='#1A1A1A', linewidth=1)
ax.add_patch(info_box)
ax.text(11, 1.2, 'LangGraph: Controle preciso sobre fluxos cíclicos de agentes baseados em Grafos de Estado.', 
        color='#EEEEEE', fontsize=8.5, fontweight='bold', ha='center')
ax.text(11, 0.8, 'Nodes (nós) representam passos de computação. Edges (arestas condicionais) controlam a lógica de fluxo com base no State.', 
        color='#B2B2B2', fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('/home/vini/ai_course/images/langgraph_concepts.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print("LangGraph concept diagram generated.")
