````markdown id="yx8p4m"
# Parte 2 — Zero-Shot Learning, Few-Shot Learning, Prompt Engineering e Self-Supervised Learning

---

# Capítulo 5 — Zero-Shot Learning

## Introdução

Até o surgimento dos grandes modelos de linguagem (Large Language Models – LLMs), praticamente todo algoritmo de Inteligência Artificial precisava ser treinado especificamente para cada tarefa que deveria executar.

Por exemplo, um modelo treinado para classificar gatos e cachorros não conseguiria reconhecer cavalos sem passar por um novo processo de treinamento utilizando milhares de imagens rotuladas.

Essa necessidade representava um enorme custo para empresas e pesquisadores, uma vez que cada novo problema exigia a construção de um novo conjunto de dados e um novo ciclo de treinamento.

Com a evolução dos modelos fundamentais (Foundation Models), surgiu um novo paradigma conhecido como **Zero-Shot Learning**, no qual o modelo consegue resolver tarefas nunca vistas anteriormente sem qualquer treinamento específico para elas.

Esse paradigma mudou profundamente a forma como profissionais de IA trabalham atualmente.

---

## Definição

Zero-Shot Learning (ZSL) consiste na capacidade de um modelo executar corretamente uma tarefa para a qual nunca recebeu exemplos rotulados durante seu treinamento específico.

Em outras palavras:

Treinamento:

```text
Perguntas sobre:
- História
- Física
- Matemática
- Medicina
- Direito
```

Nova tarefa:

```text
Traduzir um texto jurídico
```

Mesmo sem treinamento específico para tradução jurídica, um modelo suficientemente grande pode realizar essa tarefa apenas recebendo uma instrução.

---

## Como isso é possível?

A resposta está no pré-treinamento em larga escala.

Durante meses, modelos modernos aprendem padrões estatísticos presentes em:

- livros;
- artigos científicos;
- códigos-fonte;
- páginas da internet;
- documentação técnica;
- conversas;
- fóruns;
- notícias.

Ao invés de memorizar respostas, o modelo aprende representações extremamente ricas da linguagem.

Consequentemente, quando recebe uma nova instrução, consegue inferir como resolver problemas semelhantes.

---

## Exemplo

Prompt:

```text
Classifique o sentimento:

"O produto chegou antes do prazo e possui excelente qualidade."
```

Resposta esperada:

```text
Positivo
```

Nenhum treinamento adicional foi realizado.

Apenas a instrução foi suficiente.

---

## Zero-Shot em Visão Computacional

O conceito não se limita à linguagem.

Modelos como CLIP aprenderam relações entre imagens e textos.

Durante o treinamento, aprendem que:

```text
Imagem ↔ Descrição textual
```

Assim, posteriormente conseguem reconhecer objetos nunca vistos anteriormente.

Exemplo:

Prompt:

```text
Uma fotografia de um ornitorrinco.
```

Mesmo que o modelo nunca tenha sido treinado especificamente para classificar ornitorrincos, ele pode localizar imagens compatíveis utilizando o conhecimento semântico adquirido.

---

## Formulação matemática

Considere dois conjuntos.

Classes observadas:

\[
Y_s
\]

Classes nunca observadas:

\[
Y_u
\]

onde

\[
Y_s \cap Y_u = \emptyset
\]

Durante o treinamento, apenas

\[
Y_s
\]

é utilizado.

O objetivo do Zero-Shot Learning consiste em inferir corretamente elementos pertencentes a

\[
Y_u
\]

utilizando representações semânticas compartilhadas.

---

## Vantagens

Entre os principais benefícios destacam-se:

- elimina necessidade de treinamento específico;
- reduz custos computacionais;
- acelera desenvolvimento;
- aumenta reutilização de modelos;
- permite resolver problemas inéditos imediatamente.

---

## Limitações

Apesar de extremamente poderoso, o desempenho depende diretamente da qualidade do pré-treinamento.

Se o modelo nunca teve contato indireto com determinado domínio, suas respostas podem apresentar:

- alucinações;
- imprecisões;
- baixo desempenho.

---

# Capítulo 6 — Few-Shot Learning

## Motivação

Embora o Zero-Shot funcione muito bem em diversas tarefas, alguns problemas exigem exemplos para esclarecer exatamente o comportamento esperado.

Surge então o paradigma chamado **Few-Shot Learning**.

A ideia consiste em ensinar através de poucos exemplos.

---

## Definição

Few-Shot Learning é uma técnica na qual o modelo recebe alguns exemplos da tarefa antes de produzir sua resposta.

Não ocorre treinamento.

O aprendizado acontece apenas dentro do contexto da conversa.

Esse fenômeno é conhecido como **In-Context Learning**.

---

## Exemplo

Prompt:

```text
Classifique o sentimento.

Exemplo 1

Texto:
Gostei muito do atendimento.

Resposta:
Positivo

Exemplo 2

Texto:
O produto chegou quebrado.

Resposta:
Negativo

Agora classifique:

"A entrega foi extremamente rápida."

Resposta:
```

O modelo compreende o padrão desejado e responde:

```text
Positivo
```

---

## One-Shot Learning

Caso apenas um exemplo seja fornecido, utiliza-se o termo:

**One-Shot Learning**

Exemplo:

```text
Pergunta

↓

Um exemplo

↓

Nova pergunta

↓

Resposta
```

---

## N-Shot Learning

De forma geral:

- Zero-Shot → nenhum exemplo
- One-Shot → um exemplo
- Few-Shot → poucos exemplos
- Many-Shot → muitos exemplos

Embora não exista uma definição rígida, Few-Shot normalmente utiliza entre dois e vinte exemplos.

---

## Por que funciona?

Os grandes modelos desenvolveram uma capacidade surpreendente.

Eles não apenas respondem perguntas.

Eles conseguem identificar padrões temporários durante uma única interação.

Em vez de alterar seus pesos internos, utilizam o contexto disponível como memória temporária.

---

## Relação com o tamanho do contexto

Quanto maior a janela de contexto (Context Window), maior a quantidade de exemplos que pode ser utilizada.

Modelos modernos chegam a centenas de milhares ou até milhões de tokens.

Isso permite fornecer:

- manuais completos;
- código-fonte;
- contratos;
- documentação técnica;
- exemplos anteriores.

Tudo dentro da mesma conversa.

---

## Aplicações

Few-Shot tornou-se extremamente popular para:

- classificação;
- extração de entidades;
- tradução;
- geração de código;
- padronização de documentos;
- sumarização.

---

## Limitações

O principal problema é o consumo da janela de contexto.

Quanto mais exemplos forem inseridos:

- maior custo;
- maior latência;
- maior número de tokens.

Por isso surgiram técnicas para selecionar automaticamente apenas os exemplos mais relevantes.

---

# Capítulo 7 — Prompt Engineering

## Introdução

Durante os primeiros anos dos LLMs, acreditava-se que bastava escrever uma pergunta simples para obter respostas corretas.

Na prática observou-se que pequenas mudanças na forma de escrever uma instrução alteravam significativamente a qualidade das respostas.

Assim nasceu a área conhecida como **Prompt Engineering**.

---

## O que é Prompt Engineering?

Prompt Engineering consiste na criação de instruções cuidadosamente estruturadas para orientar o comportamento do modelo.

Um prompt moderno pode conter:

- contexto;
- papel (Role);
- objetivo;
- restrições;
- exemplos;
- formato da resposta;
- critérios de avaliação.

---

## Estrutura de um bom prompt

Uma estrutura bastante utilizada é:

```text
Contexto

↓

Objetivo

↓

Instruções

↓

Restrições

↓

Formato da resposta
```

---

## Exemplo simples

Prompt ruim:

```text
Explique SQL.
```

Prompt melhor:

```text
Você é professor universitário.

Explique SQL para alunos iniciantes.

Utilize exemplos.

Mostre comandos SELECT.

Apresente vantagens e limitações.

Responda em Markdown.
```

Embora ambos solicitem o mesmo tema, o segundo produz respostas significativamente superiores.

---

## Técnicas de Prompt Engineering

### Role Prompting

Define um papel para o modelo.

```text
Você é um advogado.

Você é um médico.

Você é um engenheiro.
```

---

### Chain-of-Thought Prompting

Solicita que o modelo desenvolva o raciocínio passo a passo.

Exemplo:

```text
Resolva passo a passo.
```

Essa técnica melhora tarefas matemáticas e de lógica.

---

### Self-Consistency

Executa diversos raciocínios independentes e seleciona a resposta mais consistente.

---

### Tree of Thoughts

Expande múltiplos caminhos de raciocínio simultaneamente antes de escolher o melhor.

---

### ReAct (Reason + Act)

Alterna raciocínio e utilização de ferramentas externas.

Fluxo:

```text
Pergunta

↓

Pensar

↓

Executar ferramenta

↓

Observar resultado

↓

Novo raciocínio

↓

Resposta
```

Essa abordagem tornou-se base dos agentes modernos.

---

### Prompt Templates

Em aplicações reais, prompts costumam ser parametrizados.

Exemplo:

```text
Analise o seguinte contrato:

{{documento}}

Extraia:

- partes
- valores
- prazos
- multas
```

O software substitui automaticamente as variáveis.

---

## Limitações do Prompt Engineering

Embora extremamente importante, Prompt Engineering possui limitações.

Problemas comuns incluem:

- excesso de contexto;
- instruções conflitantes;
- ambiguidades;
- respostas inconsistentes;
- dependência da ordem das informações.

Essas limitações motivaram o surgimento do **Context Engineering**, no qual o foco deixa de ser apenas o prompt e passa a ser todo o contexto disponibilizado ao modelo.

---

# Capítulo 8 — Self-Supervised Learning

## Introdução

Durante muitos anos acreditava-se que modelos somente poderiam aprender utilizando dados rotulados.

Entretanto, rotular bilhões de documentos manualmente é inviável.

Surge então um dos paradigmas mais importantes da IA moderna:

**Self-Supervised Learning (SSL).**

---

## Ideia central

Em vez de um humano fornecer os rótulos, o próprio dado gera automaticamente suas tarefas de aprendizado.

Por exemplo:

Texto original:

```text
O cachorro correu pelo parque.
```

Entrada:

```text
O cachorro [MASK] pelo parque.
```

Saída esperada:

```text
correu
```

O rótulo foi criado automaticamente.

Nenhum humano participou.

---

## Modelagem matemática

Considere uma sequência

\[
x = (x_1,x_2,\ldots,x_n)
\]

Seleciona-se uma posição \(i\).

O treinamento consiste em prever

\[
x_i
\]

utilizando todos os demais elementos.

O objetivo é minimizar a perda:

\[
L = -\log P(x_i|x_{\neq i})
\]

Esse princípio fundamenta modelos como BERT.

---

## Modelos Autoregressivos

Outra estratégia consiste em prever o próximo token.

Dada uma sequência:

```text
Hoje está fazendo muito
```

O modelo aprende a prever:

```text
calor
```

Matematicamente:

\[
P(x_t|x_1,\ldots,x_{t-1})
\]

Essa abordagem é utilizada por GPT, Llama, Qwen, Gemma, Claude e diversos outros LLMs.

---

## Benefícios

Self-Supervised Learning oferece inúmeras vantagens.

Entre elas:

- elimina necessidade de rotulação manual;
- permite utilizar toda a internet como fonte de treinamento;
- reduz custos;
- aumenta capacidade de generalização;
- produz representações altamente reutilizáveis.

---

## Foundation Models

O SSL permitiu o surgimento dos chamados **Foundation Models**.

Esses modelos são treinados uma única vez utilizando enormes quantidades de dados.

Posteriormente podem ser reutilizados para centenas de tarefas diferentes.

Essa reutilização é justamente a base do Transfer Learning, do Fine-Tuning, do Zero-Shot e do Few-Shot Learning apresentados anteriormente.

---

## A transição para a IA Generativa

O Self-Supervised Learning representa um dos maiores marcos da Inteligência Artificial moderna.

Ao permitir o treinamento em escala utilizando dados não rotulados, tornou-se possível construir modelos com centenas de bilhões de parâmetros e capacidades emergentes de raciocínio, geração de texto, programação, tradução, síntese de conhecimento e interação multimodal.

Nos capítulos seguintes será apresentado outro paradigma igualmente revolucionário: o **Reinforcement Learning**, responsável por alinhar esses modelos às preferências humanas e prepará-los para aplicações reais. Posteriormente, serão explorados conceitos como Retrieval-Augmented Generation (RAG), Agentic AI, Model Context Protocol (MCP), Harness Engineering e Loop Engineering, que representam o estado da arte no desenvolvimento de sistemas inteligentes.
````
