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

    def detect_heat_waves(self):
        """Detecta períodos de onda de calor com base nos critérios definidos."""
        # Identifica dias acima do limiar
        self.data['above_threshold'] = self.data['temp_max'] > self.data['threshold']
        # Identifica períodos consecutivos
        self.data['group'] = (self.data['above_threshold'] != self.data['above_threshold'].shift()).cumsum()
        # Agrupa periodos consecutivos
        heat_wave_candidates = self.data[self.data['above_threshold']].groupby('group')

        # Filtra apenas os periodos com duração minima
        heat_waves = []
        for name, group in heat_wave_candidates:
            if len(group) >= self.min_consecutive_days:
                start_date = group['date'].min()
                end_date = group['date'].max()
                duration = (end_date - start_date).days + 1
                max_temp = group['temp_max'].max()
                mean_temp = group['temp_max'].mean()
                intensity = group['temp_max'].sum() - group['threshold'].sum()

                # Calcula o HWMId (Heat Wave Magnitude Index daily)
                hwmid = self._calculate_hwmid(group)

                heat_waves.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': duration,
                    'max_temp': max_temp,
                    'mean_temp': mean_temp,
                    'intensity': intensity,
                    'hwmid': hwmid,
                    'year': start_date.year
                })
            self.heat_waves = pd.DataFrame(heat_waves)
            return self.heat_waves
    
    def _calculate_hwmid(self, heat_wave_data):
        """Calcula o HWMId (Heat Wave Magnitude Index daily)."""
        excess_temp = heat_wave_data['temp_max'] - heat_wave_data['threshold']
        hwmid = excess_temp.sum()
        return hwmid
    
    def generate_climate_report(self):
        """Gera um relatório climático com estatisticas das ondas de calor."""
        if self.heat_waves is None:
            self.detect_heat_waves()
        report = {
            'total_heat_waves': len(self.heat_waves),
            'avg_duration': self.heat_waves['duration'].mean(),
            'max_duration': self.heat_waves['duration'].max(),
            'avg_intesity': self.heat_waves['intensity'].mean(),
            'max_intensity': self.heat_waves['intensity'].max(),
            'avg_hwmid': self.heat_waves['hwmid'].mean(),
            'max_hwmid': self.heat_waves['hwmid'].max(),
            'annual_frequency': self._calculate_annual_frequency(),
            'decadal_trend': self._calculate_decadal_trend(),
            'monthly_distribution': self._calculate_monthly_distribution()   
        }
        self.climate_report = report
        return report
    
    def _calculate_annual_frequency(self):
        """Calcula a frequência anual de ondas de calor."""
        return self.heat_waves.groupby('year').size().reset_index(name='count')
    
    def _calculate_decadal_trend(self):
        """Calcula tendências decadais nas características das ondas de calor."""
        self.heat_waves['decade'] = self.heat_waves['year'] // 10 * 10
        trends = self.heat_waves.groupby('decade').agg({
            'duration': 'mean',
            'intensity': 'mean',
            'hwmid': 'mean',
            'year': 'count'
        }).rename(columns={
            'year': 'count'
        })
        return trends
    
    def _calculate_monthly_distribution(self):
        """Calcula a distribuição mensal das ondas de calor."""
        monthly = []
        for _, hw in self.heat_waves.iterrows():
            date_range = pd.date_range(hw['start_date'], hw['end_date'])
            months = pd.Series([d.month for d in date_range])
            monthly.extend(months.value_counts().items())
        
        monthly_df = pd.DataFrame(monthly, columns=['month', 'days'])
        monthly_df = monthly_df.groupby('month').sum().reindex(range(1,13), fill_value=0)
        monthly_df['month_name'] = monthly_df.index.map(lambda x: calendar.month_abbr[x])
        return monthly_df
    
    def plot_heat_map(self):
        """Plota um mapa de calor das ondas de calor."""
        if self.heat_waves is None:
            self.detect_heat_waves()
        
        # cria uma matriz ano x dia do ano
        years = sorted(self.data['year'].unique())
        heat_matrix = np.zeros((len(years), 366)) * np.nan

        for i, year in enumerate(years):
            year_data = self.data[self.data['year'] == year]
            heat_matrix[i, :] = year_data.set_index('day_of_year')['temp_max'].reindex(range(1, 367)).values

        # plota o mapa de calor
        plt.figure(figsize=(15, 8))
        sns.heatmap(heat_matrix, cmap='YlOrRd',
                    xticklabels=30, yticklabels=5,                  
                    cbar_kws={'label': 'Temperatura Máxima (°C)'})
        
        plt.title('Mapa de calor das ondas de calor', fontsize=20)
        plt.xlabel('Dias do ano', fontsize=16)
        plt.ylabel('Anos', fontsize=16)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_decadal_trends(self):
        """Plota graficos de tendencias decadais de ondas de calor"""
        if self.climate_report is None:
            self.generate_climate_report()
        
        trends = self.climate_report['decadal_trend']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        #Frequencia
        trends['count'].plot(ax=axes[0, 0], marker='o', color='darkred')
        axes[0,0].set_title('Frequência de Ondas de Calor por Década', fontsize=16)
        axes[0,0].set_ylabel('Numero de Ondas', fontsize=14)

        #Diração
        trends['duration'].plot(ax=axes[0, 1], marker='o', color='orangered')
        axes[0,1].set_title('Duração das Ondas de Calor por Década', fontsize=16)
        axes[0,1].set_ylabel('Dias', fontsize=14)

        #Intensidade
        trends['intensity'].plot(ax=axes[1, 0], marker='o', color='orange')
        axes[1,0].set_title('Intensidade Média por Década', fontsize=16)
        axes[1,0].set_ylabel('Temperatura (°C)', fontsize=14)

        #HWMId
        trends['hwmid'].plot(ax=axes[1, 1], marker='o', color='goldenrod')
        axes[1,1].set_title('Magnitude Média (HWMId) por Década (°C)', fontsize=16)
        axes[1,1].set_ylabel('Temperatura (°C)', fontsize=14)

        for ax in axes.flatten():
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Década', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_distribution(self):
        """Plota a distribuição mensal das ondas de calor."""
        if self.climate_report is None:
            self.generate_climate_report()
        
        monthly = self.climate_report['monthly_distribution']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='month_name', y='days', data=monthly,
                    color='orangered', order=calendar.month_abbr[1:])
        
        plt.title('Distribuição Mensal das Ondas de Calor', fontsize=16)
        plt.xlabel('Mês', fontsize=14)
        plt.ylabel('Total de Dias', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    