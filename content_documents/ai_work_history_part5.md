# Parte 5 — Agentic AI, Sistemas Multiagentes e Model Context Protocol (MCP)

---

# Capítulo 26 — O Surgimento da Agentic AI

## Introdução

Até aproximadamente 2022, a maior parte das aplicações envolvendo Large Language Models (LLMs) seguia um fluxo bastante simples:

```text
Usuário

↓

Prompt

↓

LLM

↓

Resposta
```

Esse modelo funcionava muito bem para perguntas e respostas, geração de texto, programação e tradução.

Entretanto, diversos problemas do mundo real exigem muito mais do que simplesmente gerar texto.

Considere alguns exemplos:

- desenvolver um software inteiro;
- realizar auditoria em milhares de contratos;
- navegar por dezenas de páginas Web;
- consultar bancos de dados;
- enviar e-mails;
- abrir chamados;
- acessar APIs;
- controlar robôs;
- executar comandos no sistema operacional.

Nesses casos, apenas gerar uma resposta não é suficiente.

É necessário executar ações.

Foi exatamente dessa necessidade que surgiu o paradigma conhecido como **Agentic AI**.

---

# O que é Agentic AI?

Agentic AI é um paradigma no qual modelos de linguagem deixam de atuar apenas como geradores de texto e passam a agir como agentes autônomos capazes de:

- perceber o ambiente;
- planejar ações;
- utilizar ferramentas;
- tomar decisões;
- executar tarefas;
- observar resultados;
- corrigir erros;
- repetir o processo até atingir um objetivo.

Enquanto um chatbot responde perguntas, um agente trabalha para atingir metas.

---

## Comparação

### LLM Tradicional

```text
Pergunta

↓

Resposta
```

---

### Sistema Agentic

```text
Objetivo

↓

Planejamento

↓

Ferramentas

↓

Observação

↓

Novo Planejamento

↓

Execução

↓

Resposta Final
```

Observe que existe um ciclo contínuo de tomada de decisão.

---

# O Conceito de Agente

Em Inteligência Artificial, um agente é qualquer entidade capaz de:

- perceber seu ambiente;
- interpretar informações;
- escolher ações;
- modificar o ambiente.

Formalmente:

Um agente implementa uma função

\[
f : P \rightarrow A
\]

onde

- \(P\) representa as percepções;
- \(A\) representa as ações.

Na prática moderna, um LLM tornou-se o mecanismo responsável por implementar essa função.

---

# Componentes de um Agente

Um agente moderno normalmente possui cinco componentes.

## Memória

Armazena informações relevantes.

Pode conter:

- histórico da conversa;
- documentos;
- estados anteriores;
- resultados intermediários.

---

## Planejador

Decide quais passos executar.

Exemplo:

```text
Pesquisar

↓

Ler documento

↓

Consultar banco

↓

Gerar relatório
```

---

## Ferramentas

São funções externas que podem ser utilizadas.

Exemplos:

- SQL;
- APIs;
- OCR;
- Git;
- Python;
- Navegador Web;
- Calculadora;
- Sistema Operacional.

---

## Modelo de Linguagem

Responsável pelo raciocínio.

Normalmente um LLM.

---

## Executor

Realiza efetivamente as ações escolhidas.

---

# Fluxo Geral

```text
Usuário

↓

Objetivo

↓

LLM

↓

Escolhe ferramenta

↓

Ferramenta

↓

Resultado

↓

Novo raciocínio

↓

Nova ferramenta

↓

Resposta Final
```

Esse ciclo pode ocorrer dezenas ou centenas de vezes.

---

# Ferramentas (Tools)

Uma ferramenta representa qualquer funcionalidade externa ao modelo.

Exemplo.

```python
buscar_cliente(id)
```

ou

```python
consultar_banco(sql)
```

ou

```python
executar_python()
```

O modelo não executa essas funções diretamente.

Ele apenas solicita sua execução.

Um software intermediário realiza a chamada e devolve o resultado.

---

# Function Calling

Os LLMs modernos conseguem identificar quando devem utilizar ferramentas.

Exemplo.

Usuário:

```text
Qual é o saldo do cliente João?
```

O modelo responde internamente:

```text
Chamar:

consultar_saldo(cliente="João")
```

Após receber o resultado:

```text
Saldo: R$ 8.532,40
```

Produz a resposta final.

Esse mecanismo ficou conhecido como **Function Calling**.

---

# Planejamento

Um agente raramente resolve problemas complexos em uma única etapa.

Exemplo.

Objetivo:

```text
Crie um relatório financeiro da empresa.
```

Plano possível:

1. acessar banco SQL;
2. calcular indicadores;
3. gerar gráficos;
4. escrever relatório;
5. salvar PDF.

Observe que cada etapa utiliza ferramentas diferentes.

---

# Reflexão

Uma característica importante dos agentes modernos é a capacidade de revisar suas próprias respostas.

Fluxo:

```text
Resposta

↓

Avaliação

↓

Existe erro?

↓

Sim

↓

Nova tentativa
```

Esse mecanismo reduz significativamente falhas.

---

# Capítulo 27 — Sistemas Multiagentes

## Motivação

À medida que os problemas tornam-se mais complexos, um único agente deixa de ser suficiente.

Imagine desenvolver um software completo.

São necessárias diversas especialidades:

- arquitetura;
- backend;
- frontend;
- testes;
- documentação;
- DevOps.

Em vez de criar um único agente gigantesco, pode-se criar vários agentes especializados.

---

# O que é um Sistema Multiagente?

É um conjunto de agentes que colaboram para atingir um objetivo comum.

Cada agente possui:

- responsabilidades específicas;
- ferramentas próprias;
- memória própria;
- especialização.

---

# Exemplo

```text
Supervisor

↓

──────────────────────────────

↓

Backend

Frontend

Banco

Testes

Documentação
```

Cada agente resolve apenas sua parte.

---

# Benefícios

Entre as principais vantagens:

- paralelismo;
- modularidade;
- escalabilidade;
- reutilização;
- especialização.

---

# Comunicação entre Agentes

Os agentes podem trocar mensagens.

Exemplo.

Agente Backend:

```text
API criada.
```

Agente Testes:

```text
Recebido.

Iniciando testes.
```

Agente Documentação:

```text
Atualizando manual.
```

---

# Orquestrador

Normalmente existe um agente supervisor.

Ele:

- distribui tarefas;
- monitora progresso;
- consolida respostas.

---

# Frameworks Populares

Diversos frameworks surgiram para facilitar essa arquitetura.

Entre eles:

- LangGraph;
- CrewAI;
- AutoGen;
- OpenAI Agents SDK;
- Semantic Kernel;
- Haystack Agents.

Embora diferentes, todos compartilham princípios semelhantes.

---

# Capítulo 28 — Model Context Protocol (MCP)

## Introdução

Conforme o número de ferramentas cresceu, surgiu um novo problema.

Cada ferramenta possuía uma API diferente.

Cada aplicação precisava implementar integrações específicas.

Esse cenário era semelhante ao início da internet, quando cada sistema possuía protocolos proprietários.

Era necessário criar um padrão.

Assim surgiu o **Model Context Protocol (MCP).**

---

# O que é MCP?

O Model Context Protocol é um protocolo aberto que padroniza a comunicação entre modelos de linguagem e ferramentas externas.

Ele define uma linguagem comum para que qualquer LLM possa descobrir e utilizar ferramentas sem conhecer previamente sua implementação.

Pode-se compará-lo ao papel desempenhado pelo HTTP para aplicações Web.

---

# Objetivos

O MCP busca:

- padronizar integrações;
- reduzir acoplamento;
- facilitar reutilização;
- simplificar descoberta de ferramentas;
- permitir interoperabilidade entre diferentes modelos.

---

# Arquitetura

A arquitetura básica possui três componentes.

```text
LLM

↓

Cliente MCP

↓

Servidor MCP

↓

Ferramentas
```

---

## Cliente MCP

Responsável por conversar com o modelo.

Recebe solicitações.

Encaminha chamadas.

Retorna resultados.

---

## Servidor MCP

Expõe recursos para o modelo.

Pode disponibilizar:

- ferramentas;
- prompts;
- recursos;
- documentos;
- bancos de dados.

---

## Ferramentas

São funções executáveis.

Exemplo.

```text
buscar_cliente()

executar_sql()

ler_pdf()

executar_python()
```

---

# Descoberta Automática

Uma vantagem importante do MCP é que o modelo pode descobrir automaticamente quais ferramentas existem.

Fluxo.

```text
Conectar

↓

Listar Ferramentas

↓

Selecionar Ferramenta

↓

Executar

↓

Receber Resultado
```

Não é necessário modificar o código do modelo.

---

# Recursos (Resources)

Além de ferramentas, servidores MCP podem disponibilizar recursos.

Exemplos:

- arquivos;
- documentos;
- imagens;
- bancos de conhecimento;
- configurações.

Esses recursos podem ser utilizados como contexto adicional.

---

# Prompts

Servidores MCP também podem disponibilizar prompts reutilizáveis.

Exemplo.

```text
Gerador de documentação.

Revisor jurídico.

Especialista SQL.
```

Esses prompts tornam-se ativos compartilhados.

---

# Transporte

O protocolo é independente do meio de comunicação.

Pode utilizar:

- STDIO;
- HTTP;
- WebSocket;
- Streams.

Essa flexibilidade facilita integração com aplicações locais e serviços distribuídos.

---

# Exemplo Prático

Imagine um servidor MCP especializado em leitura de PDFs.

Ferramentas disponíveis.

```text
extrair_texto()

extrair_imagens()

executar_ocr()

sumarizar_pdf()
```

Qualquer cliente compatível pode utilizá-las.

Não importa se o modelo é:

- GPT;
- Claude;
- Gemini;
- Llama;
- Qwen;
- Mistral.

Todos utilizam exatamente o mesmo protocolo.

---

# Benefícios

O MCP oferece inúmeras vantagens.

- desacoplamento;
- reutilização;
- padronização;
- interoperabilidade;
- manutenção simplificada;
- expansão modular.

Por isso tornou-se rapidamente um dos principais padrões para construção de agentes inteligentes.

---

# Relação entre Agentic AI e MCP

É importante compreender que MCP não substitui os agentes.

Ele fornece a infraestrutura para que agentes utilizem ferramentas de forma padronizada.

Podemos resumir a relação da seguinte forma:

```text
Agentic AI

↓

Precisa acessar ferramentas

↓

MCP padroniza esse acesso

↓

Ferramentas executam ações

↓

Resultados retornam ao agente
```

Essa separação reduz fortemente o acoplamento entre modelos e aplicações.

---

# Evolução do Trabalho em IA

Durante muitos anos, o profissional de IA preocupava-se principalmente em treinar modelos.

Posteriormente, passou a realizar Fine-Tuning.

Em seguida, dedicou-se ao Prompt Engineering.

Com a popularização do RAG, tornou-se necessário projetar pipelines completos de recuperação de conhecimento.

Hoje, entretanto, o foco mudou novamente.

Grande parte do trabalho consiste em construir agentes capazes de colaborar entre si, acessar ferramentas externas, utilizar protocolos como MCP e resolver tarefas complexas de maneira autônoma.

Essa mudança representa uma nova etapa na evolução da Inteligência Artificial, aproximando os modelos de sistemas capazes de atuar continuamente em ambientes reais. Na próxima parte serão apresentados os conceitos de **Harness Engineering** e **Loop Engineering**, considerados atualmente um dos estados mais avançados da engenharia de sistemas baseados em IA.