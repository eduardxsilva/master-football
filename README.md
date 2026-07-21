# Football Analytics Engine — Copa 2026 e Multicampeonatos

Aplicação web em Streamlit para análise estatística, previsão de partidas e simulação de competições. Além da Copa do Mundo de 2026, permite selecionar campeonatos de clubes, baixar resultados e calendário e prever as próximas partidas.

O projeto possui interface desenvolvida em Streamlit e reúne modelos de Elo dinâmico, Poisson, correção Dixon-Coles, aprendizado de máquina e ensemble. Métricas de equipes obtidas da FIFA e odds de mercado podem ser incorporadas como dados complementares.

> Este é um projeto independente, sem vínculo oficial com a FIFA. As previsões são estimativas probabilísticas e não garantem resultados reais. O aplicativo não deve ser tratado como recomendação de aposta.

## Funcionalidades

- Importação de uma base histórica em CSV ou Excel;
- seleção de Brasileirão A/B, Libertadores, Premier League, Championship, La Liga, Serie A, Bundesliga, Ligue 1 e outras competições;
- coleta de jogos concluídos e calendário futuro pela API-Football;
- previsões em lote para as partidas futuras do campeonato selecionado;
- leitura de colunas equivalentes em português ou inglês;
- importação de um Excel consolidado com histórico, estado da Copa e métricas FIFA;
- extração alternativa de dados históricos da internet;
- cálculo e visualização do ranking Elo;
- previsão de confrontos entre duas equipes;
- probabilidades de vitória, empate e derrota;
- projeção de placares por modelos de Poisson;
- correção Dixon-Coles para resultados de baixa contagem;
- modelos de Machine Learning com `scikit-learn`;
- combinação de modelos por ensemble;
- integração opcional com odds de mercado;
- simulação Monte Carlo de mata-mata;
- simulação aproximada do formato da Copa de 2026;
- backtest temporal para avaliar os modelos sem usar jogos futuros no treinamento;
- exportação dos resultados em um relatório Excel com várias abas.

## Modelos utilizados

O motor analítico implementa diferentes abordagens:

- **Elo dinâmico:** atualiza a força relativa das equipes ao longo do histórico;
- **Poisson:** estima gols esperados e probabilidades de placares;
- **Dixon-Coles:** ajusta a distribuição de resultados com poucos gols;
- **Poisson bivariada:** considera dependência entre os gols das equipes;
- **Machine Learning:** utiliza modelos como regressão logística, Random Forest e HistGradientBoosting;
- **Ensemble:** combina os resultados dos modelos estatísticos e de aprendizado de máquina;
- **Monte Carlo:** repete o torneio milhares de vezes para estimar chances de avanço e título.

As métricas FIFA, quando presentes no arquivo importado, são transformadas em índices de ataque, defesa, goleiro, distribuição, disciplina, condição física e desempenho geral. Sem essas métricas, o aplicativo continua funcionando com o histórico de partidas.

## Estrutura do projeto

```text
FIFA2026_MAE/
├── .streamlit/
│   └── config.toml       # Tema e configuração visual do Streamlit
├── fifa2026_core.py      # Extração, tratamento, modelos e simuladores
├── competition_data.py   # Coleta e calendário multicompetições
├── streamlit_app.py      # Interface web e fluxo da aplicação
├── gerador_excel_fifa2026_edge_v5.py
│                          # Gerador local de dados via Edge/Selenium
├── corrigir_excel_fifa2026_standings.py
│                          # Correção de standings em Excel já gerado
├── requirements.txt      # Dependências Python
└── README.md
```

## Requisitos

- Python 3.10 ou superior;
- `pip`;
- conexão com a internet para extração online ou consulta de odds.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/eduardxsilva/FIFA2026_MAE.git
cd FIFA2026_MAE
```

Crie e ative um ambiente virtual.

No Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

Inicie a interface:

```bash
streamlit run streamlit_app.py
```

O Streamlit exibirá no terminal o endereço local da aplicação, normalmente `http://localhost:8501`.

## Publicar no GitHub e Streamlit Community Cloud

Crie um repositório vazio no GitHub e, dentro da pasta extraída, execute:

```bash
git init
git add .
git commit -m "Aplicativo Streamlit multicompetições"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Depois:

1. Acesse `share.streamlit.io` e conecte sua conta do GitHub.
2. Selecione o repositório, a branch `main` e o arquivo `streamlit_app.py`.
3. Em **Advanced settings > Secrets**, adicione:

```toml
API_FOOTBALL_KEY = "sua_chave_privada"
ODDS_API_KEY = "sua_chave_opcional"
```

4. Faça o deploy. Não envie `.streamlit/secrets.toml` ao GitHub.

## Campeonatos e API-Football

Na página **Campeonatos**, selecione a competição e o provedor. O modo TheSportsDB gratuito baixa automaticamente as duas temporadas recentes: 2025 e 2026 para competições de ano-calendário, ou 2025–2026 e 2026–2027 para ligas europeias. A aplicação une os jogos encerrados para treinamento e separa o calendário futuro da temporada mais nova. Depois do treinamento, o botão **Prever calendário futuro** gera probabilidades 1X2, xG, placar provável, mais de 2,5 gols e ambas marcam.

A temporada é o ano inicial informado pelo provedor. A disponibilidade de histórico, calendário e competições depende do plano da API. A chave pode ser digitada na sessão ou configurada como `API_FOOTBALL_KEY` nos Secrets.

TheSportsDB é uma base colaborativa: algumas competições podem ter partidas ausentes ou não possuir ID gratuito confirmado. O aplicativo informa quantos jogos foram efetivamente encontrados e não cria resultados inexistentes. A API-Football continua disponível como provedor alternativo para chaves com acesso às temporadas atuais.

## Formato mínimo dos dados

A base histórica deve conter uma linha por partida e as seguintes colunas:

| Inglês | Equivalente em português | Descrição |
| --- | --- | --- |
| `date` | `data` | Data da partida |
| `home_team` | `mandante` | Equipe mandante ou Time A |
| `away_team` | `visitante` | Equipe visitante ou Time B |
| `home_goals` | `gols_casa` | Gols do mandante |
| `away_goals` | `gols_fora` | Gols do visitante |

Colunas opcionais:

| Inglês | Equivalente em português | Descrição |
| --- | --- | --- |
| `home_xg` | `xg_casa` | Gols esperados do mandante |
| `away_xg` | `xg_fora` | Gols esperados do visitante |
| `competition` | `competicao` | Competição da partida |

O formato recomendado é um único arquivo Excel contendo a base histórica e, quando disponíveis, abas como `relatorio_mestre_equipes` ou `team_stats_*`. Essas abas permitem incorporar métricas FIFA ao treinamento e às simulações.

## Fluxo de uso

1. Acesse **Campeonatos** para baixar uma liga ou **Importar dados** para carregar CSV/XLSX.
2. Verifique no dashboard se partidas e equipes foram reconhecidas corretamente.
3. Acesse **Treinar modelos** para preparar Elo, Poisson, Machine Learning e ensemble.
4. Em **Prever partida**, selecione as equipes e calcule a previsão.
5. Em **Simulações**, execute o mata-mata ou a simulação aproximada da Copa de 2026.
6. Em **Validação**, rode o backtest temporal e compare as métricas dos modelos.
7. Em **Exportar**, baixe o relatório `relatorio_fifa2026_analytics.xlsx`.

## Odds de mercado — opcional

O aplicativo aceita integração com `odds-api.io` e `The Odds API`. A chave pode ser digitada na interface ou configurada com o nome `ODDS_API_KEY`.

Por variável de ambiente:

```bash
export ODDS_API_KEY="sua_chave"
```

No Windows PowerShell:

```powershell
$env:ODDS_API_KEY="sua_chave"
```

Em uma implantação Streamlit, a chave também pode ser definida nos secrets da plataforma:

```toml
ODDS_API_KEY = "sua_chave"
```

Nunca publique chaves no código, no README, no histórico do Git ou em arquivos versionados. As odds funcionam como leitura complementar do mercado; o placar provável continua sendo produzido pelo modelo estatístico.

O arquivo `.streamlit/secrets.toml`, planilhas, logs e saídas de depuração estão
incluídos no `.gitignore`. Mesmo assim, revise o conteúdo preparado com
`git status` antes de cada commit.

## Validação

O backtest segue ordem temporal: os modelos são treinados com partidas anteriores e avaliados em partidas posteriores. O relatório apresenta, conforme a disponibilidade dos dados:

- acurácia;
- log loss;
- Brier Score;
- número de partidas avaliadas;
- previsões individuais do conjunto de teste.

Resultados de validação dependem diretamente da cobertura, consistência e atualidade da base. Uma boa acurácia histórica não prova desempenho futuro.

## Tecnologias

- Python;
- Streamlit;
- pandas e NumPy;
- SciPy;
- scikit-learn;
- Plotly;
- Beautiful Soup e Requests;
- OpenPyXL.

## Limitações

- O desempenho depende da qualidade e da quantidade dos jogos importados;
- escalações, lesões, suspensões e decisões táticas podem não estar representadas;
- dados extraídos de páginas externas podem deixar de funcionar quando os sites mudarem;
- APIs de odds exigem chave própria e podem possuir limites ou custos;
- a simulação da Copa de 2026 é uma aproximação baseada nos dados e na configuração disponíveis;
- probabilidades não devem ser interpretadas como certeza ou promessa de retorno financeiro.

## Licença

Este repositório ainda não contém um arquivo de licença. Enquanto uma licença não for adicionada, não se deve presumir permissão para copiar, modificar ou redistribuir o código.
