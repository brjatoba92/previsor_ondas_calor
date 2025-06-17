🌡️ Heat Wave Analyzer: Análise Avançada de Ondas de Calor

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Pandas-1.3%252B-orange
https://img.shields.io/badge/Matplotlib-3.4%252B-blue
https://img.shields.io/badge/Seaborn-0.11%252B-red
https://img.shields.io/badge/SciPy-1.7%252B-blueviolet

O Heat Wave Analyzer é uma solução científica completa para detecção, análise e visualização de ondas de calor em séries temporais climáticas. Desenvolvido para pesquisadores, meteorologistas e cientistas de dados, este pacote oferece ferramentas robustas para estudar eventos de calor extremo com rigor acadêmico.

📌 Índice

    Funcionalidades Principais

    Instalação

    Como Usar

    Métricas Científicas

    Visualizações

    Estrutura do Projeto

    Exemplo Prático

    Contribuição

    Licença

    Contato


🌟 Funcionalidades Principais
🔍 Detecção Avançada de Ondas de Calor

    Algoritmo baseado em percentis móveis (janela de 15 dias)

    Limiares dinâmicos por dia do ano

    Critérios personalizáveis (duração mínima, percentil de threshold)

📊 Análise Quantitativa

    HWMId (Heat Wave Magnitude Index daily) - padrão científico

    Intensidade acumulada e temperatura máxima

    Duração média e máxima dos eventos

    Frequência anual e tendências decadais

📈 Visualização Profissional

    Mapas de calor temporais interativos

    Gráficos de tendência por década

    Distribuição mensal dos eventos

    Estilo visual profissional pronto para publicações

💾 Exportação de Resultados

    Relatórios completos em CSV

    Figuras em alta resolução (PNG)

    Estrutura organizada em diretórios

⚙️ Instalação
Pré-requisitos

    Python 3.8 ou superior

    pip (gerenciador de pacotes Python)

Passo a Passo

    Clone o repositório:

bash

git clone https://github.com/brjatoba92/previsor_ondas_calor.git
cd heat-wave-analyzer
    Crie um ambiente virtual (recomendado):

bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

    Instale as dependências:

bash

pip install -r requirements.txt

🚀 Como Usar
Estrutura Básica dos Dados

Seu arquivo de dados deve conter:

    Coluna date (datetime): datas das observações

    Coluna temp_max (float): temperaturas máximas diárias

Exemplo mínimo:
csv

date,temp_max
1980-01-01,28.5
1980-01-02,29.1
...

Código Básico
python

import pandas as pd
from heatwave_analyzer import HeatWaveAnalyzer

# 1. Carregar dados
data = pd.read_csv('dados_climaticos.csv')
data['date'] = pd.to_datetime(data['date'])

# 2. Criar analisador (parâmetros padrão científicos)
analyzer = HeatWaveAnalyzer(
    data,
    threshold_percentile=90,  # percentil para limiar
    min_consecutive_days=3    # duração mínima da onda
)

# 3. Detectar ondas de calor
heat_waves = analyzer.detect_heat_waves()

# 4. Gerar relatório completo
report = analyzer.generate_climate_report()

# 5. Exportar resultados
analyzer.save_climate_report_to_csv('resultados/relatorios')
analyzer.plot_heat_map('resultados/graficos/mapa_calor.png')

🔬 Métricas Científicas
📏 HWMId (Heat Wave Magnitude Index daily)

Métrica padrão na literatura científica calculada como:
text

HWMId = ∑(Tmax - Tthreshold) para todos os dias do evento

Onde:

    Tmax: temperatura máxima observada

    Tthreshold: limiar de temperatura para o dia do ano

📈 Outras Métricas Calculadas
Métrica	Descrição
Duração	Número de dias consecutivos acima do limiar
Intensidade	Soma acumulada do excesso de temperatura (Tmax - Tthreshold)
Temperatura Máxima	Valor máximo observado durante o evento
Temperatura Média	Média das temperaturas máximas durante o evento
Frequência Anual	Número de eventos por ano
Tendência Decadal	Evolução das características ao longo de períodos de 10 anos

📊 Visualizações
1. Mapa de Calor Temporal

https://via.placeholder.com/600x300?text=Heat+Map+Example

Visualização das temperaturas máximas ao longo dos anos, com destaque para os períodos de onda de calor.
2. Tendências Decadais

https://via.placeholder.com/600x300?text=Trend+Analysis

Série de gráficos mostrando a evolução de:

    Frequência de eventos

    Duração média

    Intensidade média

    Magnitude (HWMId)

3. Distribuição Mensal

https://via.placeholder.com/600x300?text=Monthly+Distribution

Barras verticais mostrando em quais meses ocorrem mais dias de onda de calor.
📁 Estrutura do Projeto
text

heat-wave-analyzer/
│
├── heatwave_analyzer.py       # Classe principal com toda a lógica
├── requirements.txt           # Dependências do projeto
├── README.md                  # Este arquivo
├── dados/                     # Pasta para dados de entrada (opcional)
│   └── exemplo_clima.csv      
└── resultados/                # Pasta gerada automaticamente
    ├── relatorios/            # Relatórios em CSV
    │   ├── annual_frequency.csv
    │   ├── climate_report_summary.csv
    │   └── ...
    └── graficos/              # Visualizações exportadas
        ├── mapa_calor.png
        └── ...

🔍 Exemplo Prático
Análise de Dados Climáticos de 40 Anos
python

# Configuração avançada
analyzer = HeatWaveAnalyzer(
    data,
    threshold_percentile=92,  # Limiar mais rigoroso
    min_consecutive_days=4    # Eventos mais prolongados
)

# Análise completa
heat_waves = analyzer.detect_heat_waves()
report = analyzer.generate_climate_report()

# Exportação organizada
analyzer.save_climate_report_to_csv('resultados/relatorio_avancado')
analyzer.plot_decadal_trends('resultados/graficos/tendencias_decadais.png')

Interpretando os Resultados

    Relatório Sumário (climate_report_summary.csv):

        Total de eventos detectados

        Métricas médias e máximas

    Frequência Anual (annual_frequency.csv):

        Evolução do número de eventos por ano

        Identificação de anos mais críticos

    Tendências Decadais (decadal_trends.csv):

        Mudanças nas características das ondas de calor ao longo do tempo

        Evidências de aumento de intensidade ou frequência

🤝 Contribuição

Contribuições são bem-vindas! Siga estes passos:

    Faça um fork do projeto

    Crie uma branch para sua feature (git checkout -b feature/incrivel)

    Commit suas mudanças (git commit -m 'Adiciona feature incrível')

    Push para a branch (git push origin feature/incrivel)

    Abra um Pull Request

📜 Licença

Distribuído sob a licença MIT. Veja LICENSE para mais informações.
✉️ Contato

Para dúvidas científicas ou suporte técnico:

    Email: [brunojatobadev@gmail.com]

    Issues: [https://github.com/brjatoba92/previsor_ondas_calor]