# Importação das dependencias necessarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import calendar
from datetime import datetime, timedelta

# Configurações Iniciais
plt.style.use('ggplot')
sns.set_palette("YlOrRd")

class HeatWaveAnalyzer:
    """
        Inicializa o analisador de ondas de calor.
        
        Args:
            temp_data (DataFrame): Dados de temperatura com colunas 'date' e 'temp_max'
            threshold_percentile (int): Percentil para definir o limiar de onda de calor (padrão: 90)
            min_consecutive_days (int): Mínimo de dias consecutivos para considerar onda de calor (padrão: 3)
        """
    def __init__(self, temp_data, threshold_percentile=90, min_consecutive_days=3):
        self.data = temp_data.copy()
        self.threshold_percentile = threshold_percentile
        self.min_consecutive_days = min_consecutive_days
        self.heatwaves = None
        self.climate_report = None

        # Pre-processamento
        self._preprocess_data()
    
    def _preprocess_data(self):
        # Formata corretamente a coluna 'date'
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

        # Colunas auxiliares
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['day_of_year'] = self.data['date'].dt.dayofyear

        # Calcula o percentil movel para definir o limiar de temperatura
        self._calculate_thresholds()
    
    def _calculate_thresholds(self, window_size=15):
        """
        Calcula o limiar de temperatura para cada dia do ano usando uma janela móvel.
        
        Args:
            window_size (int): Tamanho da janela em dias para calcular o percentil (padrão: 15)
        """
        thresholds = []

        for day in range(1, 367): # para cada dia do ano
            
            # Define a janela ao redor do dia do ano (considerando anos bissextos)
            window_start = max(1, day - window_size)
            window_end = min(366, day + window_size)

            # Filtra os dados da janela
            window_data = self.data[
                (self.data['day_of_year'] >= window_start) & 
                (self.data['day_of_year'] <= window_end)
            ]['temp_max']

            if not window_data.empty:
                threshold = np.percentile(window_data, self.threshold_percentile)
                thresholds.append(threshold)
            else:
                thresholds.append(np.nan)
        
        # Cria um DataFrame com os limiares por dia do ano
        self.thresholds = pd.DataFrame({
            'day_of_year': range(1, 367),
            'threshold': thresholds
        })

        # Merge com os dados originais
        self.data = pd.merge(self.data, self.thresholds, on='day_of_year', how='left')

