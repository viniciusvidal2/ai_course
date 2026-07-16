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
ax.text(11, 11, 'Arquitetura do Ecossistema LangChain', color='#EEEEEE', fontsize=14, fontweight='bold', ha='center')

# Left section: Core Components
ax.text(5, 9.5, 'Componentes do Core', color='#00ADB5', fontsize=11, fontweight='bold', ha='center')

# Draw boxes for core components
components = [
    ('Prompt Templates', 'Estruturação dinâmica de prompts', 8),
    ('Modelos I/O (LLMs/Chat)', 'Interface unificada para modelos', 6),
    ('Output Parsers', 'Conversão de saída em tipos estruturados', 4),
    ('Retrieval (RAG)', 'Loaders, Splitters, VectorStores', 2)
]

for name, desc, y in components:
    box = patches.FancyBboxPatch((1.5, y-0.6), 7, 1.2, boxstyle="round,pad=0.1", edgecolor='#393E46', facecolor='#222831', linewidth=1)
    ax.add_patch(box)
    ax.text(5, y+0.1, name, color='#00ADB5', fontsize=9, fontweight='bold', ha='center')
    ax.text(5, y-0.3, desc, color='#B2B2B2', fontsize=7.5, ha='center')

# Flow Arrow to center
ax.annotate('', xy=(10.5, 5.5), xytext=(9.2, 5.5), arrowprops=dict(arrowstyle="->", color='#EEEEEE', lw=2))

# Right section: Composition & Pipelines
ax.text(16, 9.5, 'Composição e Pipelines (LCEL)', color='#FF2E63', fontsize=11, fontweight='bold', ha='center')

# Chain flow diagram
box_chain = patches.FancyBboxPatch((12.5, 4.0), 7.5, 4.5, boxstyle="round,pad=0.2", edgecolor='#FF2E63', facecolor='#1E111A', linewidth=1.5)
ax.add_patch(box_chain)

# Chain blocks inside
chain_steps = [
    ('Template', 16.25, 7.3),
    ('LLM / Model', 16.25, 6.0),
    ('Parser', 16.25, 4.7)
]

for step_name, x, y in chain_steps:
    step_box = patches.FancyBboxPatch((x-2, y-0.4), 4, 0.8, boxstyle="round,pad=0.1", edgecolor='#393E46', facecolor='#303841', linewidth=1)
    ax.add_patch(step_box)
    ax.text(x, y-0.1, step_name, color='#EEEEEE', fontsize=8.5, fontweight='bold', ha='center')

# Arrows between steps
ax.annotate('', xy=(16.25, 6.5), xytext=(16.25, 6.8), arrowprops=dict(arrowstyle="<-", color='#FF2E63', lw=1.5))
ax.annotate('', xy=(16.25, 5.2), xytext=(16.25, 5.5), arrowprops=dict(arrowstyle="<-", color='#FF2E63', lw=1.5))

# LCEL Pipeline Text
ax.text(16.25, 3.3, 'Sintaxe LCEL (LangChain Expression Language):\nchain = template | model | parser', 
        color='#EEEEEE', fontsize=8.5, fontweight='bold', ha='center', style='italic')

# Bottom explanation
info_box = patches.FancyBboxPatch((1.5, 0.6), 18.5, 1.2, boxstyle="round,pad=0.1", edgecolor='#393E46', facecolor='#1A1A1A', linewidth=1)
ax.add_patch(info_box)
ax.text(10.75, 1.2, 'Vantagem do LangChain: Padronização de componentes para criação ágil de aplicações baseadas em LLM.', 
        color='#EEEEEE', fontsize=8.5, fontweight='bold', ha='center')
ax.text(10.75, 0.8, 'Permite trocar de provedor (ex: OpenAI para Ollama local) alterando apenas uma linha de código.', 
        color='#B2B2B2', fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('/home/vini/ai_course/images/langchain_architecture.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print("LangChain architecture diagram generated.")
