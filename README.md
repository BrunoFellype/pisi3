# 📊  ASTRA — Análise de Dados

Este repositório contém a etapa de análise de dados do projeto
**ASTRA — Ambiente de Suporte à Trajetória e Rotina Acadêmica**.

A análise utiliza um conjunto de dados sobre hábitos, características
e desempenho de estudantes universitários, buscando identificar padrões,
relações e possíveis perfis presentes nos dados.

> 🔗 Este repositório faz parte do projeto ASTRA.
> Para conhecer a aplicação mobile, acesse o
> [ASTRA - Mobile](https://github.com/brunoFellype/dsi).

## Sobre o projeto


## 🎯 Objetivos

A análise de dados tem como objetivos:
- Compreender as características do conjunto de dados;
- Explorar a distribuição das variáveis;
- Investigar associações entre hábitos e desempenho acadêmico;
- Visualizar relações relevantes entre as variáveis;
- Identificar possíveis agrupamentos de estudantes por meio de K-Means;
- Avaliar modelos de Machine Learning para previsão do GPA.

## 🗃️ Dataset

O conjunto de dados utilizado contém **10.000 registros de estudantes
universitários e 27 variáveis**, abrangendo informações relacionadas a
características dos estudantes, hábitos, contexto acadêmico e desempenho.

Entre as variáveis analisadas estão:

- horas de estudo por dia;
- frequência nas aulas;
- horas de sono;
- tempo de tela;
- uso de redes sociais;
- horas de jogos;
- horas de exercício;
- nível de estresse;
- uso de ferramentas de IA;
- dias de preparação para exames;
- consumo de café;
- atividades extracurriculares;
- GPA;
- nota do exame final;
- nota das atividades;

## 🔬 Etapas da análise

A análise de dados é organizada nas seguintes etapas:

### 1. Entendimento dos dados

Inicialmente são analisadas a estrutura do dataset, os tipos das variáveis,
valores ausentes, duplicidades e distribuição das categorias.

### 2. Análise Exploratória de Dados (EDA)

São utilizadas estatísticas descritivas e visualizações para compreender
a distribuição das variáveis e as características gerais dos estudantes.

### 3. Análise de correlações

São investigadas associações lineares entre variáveis relacionadas aos
hábitos dos estudantes e diferentes medidas de desempenho acadêmico,
como GPA, nota do exame final e nota das atividades.

### 4. Visualizações

Os resultados são apresentados por meio de gráficos interativos,
permitindo explorar as relações encontradas nos dados.

### 5. Machine Learning

São utilizadas técnicas de Machine Learning para complementar a análise.

#### K-Means

Utilizado para identificar agrupamentos de estudantes com características
semelhantes.

#### Modelo preditivo

São avaliados modelos de regressão para investigar a possibilidade de
prever o GPA a partir de características selecionadas dos estudantes.

## Tecnologias utilizadas
<div align=center>
    <img width='100' src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original-wordmark.svg" />
    <img width='100' src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-plain-wordmark.svg" alt="Pandas"/>
    <img width='100' src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/plotly/plotly-original.svg" />
    <img width='100' src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original-wordmark.svg" />
    <img width='100' src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" />
</div> 

## 🚀 Como executar
### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
cd ASTRA_Dashboard
```

### 2. Crie um ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o Dashboard
```bash
streamlit run app.py
```
> Após a execução, o Streamlit disponibilizará o dashboard localmente no navegador.

## Relação com o ASTRA
A análise de dados constitui uma das etapas do projeto ASTRA e está
relacionada ao desenvolvimento da aplicação mobile.

```text
┌─────────────────────────┐
│      Projeto ASTRA      │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌───────────┐  ┌───────────────┐
│ Aplicação │  │ Análise Dados │
│   Mobile  │  │   + Dashboard │
└───────────┘  └───────────────┘
                    │
                    ├── EDA
                    ├── Correlações
                    ├── Visualizações
                    ├── K-Means
                    └── Predição
```