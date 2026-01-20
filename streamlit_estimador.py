"""
Estimador de Esfuerzo - Payroll Diecisiete
==========================================
App Streamlit para estimación de bugs

Autor: Diego Flores
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy import stats
from typing import Dict

# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================

st.set_page_config(
    page_title="Estimador Diecisiete",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASE ESTIMADOR (copiada de la notebook)
# ============================================================

class EstimadorDiecisiete:
    """Sistema de estimación robusto para bugs de Diecisiete."""
    
    def __init__(self, df_historico: pd.DataFrame):
        self.df_historico = df_historico
        self.stats_cliente = self._calcular_stats_cliente()
        self.global_median = df_historico['worklog_hours'].median()
        self.global_mae = self._calcular_mae_global()
        self.outlier_threshold = 20.0
    
    def _calcular_stats_cliente(self) -> Dict:
        stats_dict = {}
        for cliente in self.df_historico['cliente'].unique():
            data = self.df_historico[
                self.df_historico['cliente'] == cliente
            ]['worklog_hours']
            
            if len(data) >= 3:
                median = data.median()
                mae = np.mean(np.abs(data - median))
                stats_dict[cliente] = {
                    'median': median,
                    'mae': mae,
                    'count': len(data),
                    'p25': data.quantile(0.25),
                    'p75': data.quantile(0.75),
                }
            else:
                stats_dict[cliente] = None
        return stats_dict
    
    def _calcular_mae_global(self) -> float:
        median_global = self.df_historico['worklog_hours'].median()
        return np.mean(np.abs(self.df_historico['worklog_hours'] - median_global))
    
    def _extraer_keywords(self, texto: str) -> Dict[str, int]:
        texto_lower = texto.lower()
        return {
            'is_irpf': 1 if re.search(r'\birpf\b', texto_lower) else 0,
            'is_ajuste_irpf': 1 if re.search(r'ajuste.*irpf|correc.*irpf', texto_lower) else 0,
            'is_licencias': 1 if re.search(r'\blicenc|\bpermiso|\bvacaci', texto_lower) else 0,
            'is_planillas': 1 if re.search(r'\bplanilla|\bmasiv', texto_lower) else 0,
            'is_aguinaldo': 1 if re.search(r'\baguinaldo\b', texto_lower) else 0,
        }
    
    def _aplicar_reglas(self, base: float, cliente: str, keywords: Dict, 
                       texto_len: int) -> tuple:
        multiplicador = 1.0
        factores = []
        
        if cliente == "Intendencia de Rivera":
            multiplicador *= 1.15
            factores.append("Cliente Rivera (+15%)")
        
        if keywords['is_irpf']:
            multiplicador *= 1.25
            factores.append("IRPF (+25%)")
        
        if keywords['is_ajuste_irpf']:
            multiplicador *= 1.35
            factores.append("⚠️ Ajuste IRPF (+35%)")
        
        if keywords['is_licencias']:
            multiplicador *= 1.15
            factores.append("Licencias (+15%)")
        
        if keywords['is_planillas']:
            multiplicador *= 1.30
            factores.append("Planillas/Masivo (+30%)")
        
        if keywords['is_aguinaldo']:
            multiplicador *= 1.20
            factores.append("Aguinaldo (+20%)")
        
        if texto_len > 150:
            multiplicador *= 1.20
            factores.append("Descripción muy extensa (+20%)")
        elif texto_len > 80:
            multiplicador *= 1.10
            factores.append("Descripción extensa (+10%)")
        
        if keywords['is_irpf'] and keywords['is_planillas']:
            multiplicador *= 1.25
            factores.append("🚨 IRPF + Masivo (+25% adicional)")
        
        if keywords['is_ajuste_irpf'] and keywords['is_planillas']:
            multiplicador *= 1.40
            factores.append("🚨🚨 Ajuste IRPF + Masivo (+40% adicional)")
        
        return base * multiplicador, factores
    
    def predecir(self, cliente: str, summary: str, description: str = "") -> Dict:
        cliente_stats = self.stats_cliente.get(cliente)
        
        if cliente_stats and cliente_stats['count'] >= 5:
            base = cliente_stats['median']
            mae = cliente_stats['mae']
            confianza = "Alta"
            metodo = f"Mediana cliente ({cliente_stats['count']} casos)"
        elif cliente_stats and cliente_stats['count'] >= 3:
            base = (cliente_stats['median'] + self.global_median) / 2
            mae = (cliente_stats['mae'] + self.global_mae) / 2
            confianza = "Media"
            metodo = f"Híbrido ({cliente_stats['count']} casos)"
        else:
            base = self.global_median
            mae = self.global_mae
            confianza = "Baja"
            metodo = "Mediana global (sin historial)"
        
        texto_completo = f"{summary} {description}"
        keywords = self._extraer_keywords(texto_completo)
        texto_len = len(texto_completo.split())
        
        pred_central, factores = self._aplicar_reglas(
            base, cliente, keywords, texto_len
        )
        
        pred_min = max(0.5, pred_central - mae)
        pred_max = pred_central + mae
        
        pred_central = round(pred_central * 4) / 4
        pred_min = round(pred_min * 4) / 4
        pred_max = round(pred_max * 4) / 4
        
        if pred_central > 15:
            factores.append("🚨 Estimación >15h - considerar spike técnico")
        
        if confianza == "Baja":
            factores.append("⚠️ Cliente sin historial - rango amplio")
        
        if pred_central >= self.outlier_threshold * 0.8:
            factores.append("⚠️ Posible outlier - revisar complejidad")
        
        return {
            'horas_centrales': pred_central,
            'intervalo_min': pred_min,
            'intervalo_max': pred_max,
            'confianza': confianza,
            'factores': factores,
            'metodo': metodo,
            'casos_historicos': cliente_stats['count'] if cliente_stats else 0,
            'base_sin_ajustar': round(base * 4) / 4,
        }

# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_estimador():
    """Carga el estimador desde CSV histórico."""
    try:
        # Leer dataset histórico
        df_historico = pd.read_csv('historico_bugs.csv')
        
        # Crear estimador
        estimador = EstimadorDiecisiete(df_historico)
        return estimador
        
    except FileNotFoundError:
        st.error("⚠️ Archivo historico_bugs.csv no encontrado")
        st.info("Archivos disponibles: " + str(Path('.').glob('*')))
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

estimador = cargar_estimador()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🎯 Estimador de Esfuerzo - Payroll Diecisiete</h1>
    <p>Sistema basado en estadísticas robustas + reglas de negocio calibradas</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FORMULARIO
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Información del Bug")
    
    # Cliente
    clientes = sorted(estimador.stats_cliente.keys())
    cliente = st.selectbox(
        "Cliente *",
        options=clientes,
        help="Selecciona el cliente afectado"
    )
    
    # Resumen
    summary = st.text_input(
        "Resumen del bug *",
        placeholder="Ej: Error en cálculo de IRPF en aguinaldo",
        help="Breve descripción del problema"
    )

with col2:
    st.subheader("📝 Descripción Detallada")
    
    description = st.text_area(
        "Descripción (opcional)",
        height=150,
        placeholder="Pasos para reproducir, comportamiento esperado, screenshots...",
        help="Información adicional sobre el bug"
    )

# Botón estimar
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

with col_btn1:
    btn_estimate = st.button("🔮 Estimar Esfuerzo", type="primary", use_container_width=True)

with col_btn2:
    btn_clear = st.button("🔄 Limpiar", use_container_width=True)

# ============================================================
# LÓGICA DE ESTIMACIÓN
# ============================================================

if btn_clear:
    st.rerun()

if btn_estimate:
    if not summary:
        st.warning("⚠️ Por favor ingresa un resumen del bug")
    else:
        # Predecir
        pred = estimador.predecir(
            cliente=cliente,
            summary=summary,
            description=description
        )
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📊 Resultado de la Estimación")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Horas Estimadas",
                value=f"{pred['horas_centrales']}h",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Rango Mínimo",
                value=f"{pred['intervalo_min']}h",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Rango Máximo",
                value=f"{pred['intervalo_max']}h",
                delta=None
            )
        
        with col4:
            confianza_emoji = {
                'Alta': '✅',
                'Media': '⚠️',
                'Baja': '❌'
            }
            st.metric(
                label="Confianza",
                value=f"{confianza_emoji.get(pred['confianza'], '')} {pred['confianza']}",
                delta=None
            )
        
        # Detalles
        st.markdown("---")
        col_det1, col_det2 = st.columns(2)
        
        with col_det1:
            st.markdown("### 📐 Cálculo")
            st.markdown(f"""
            - **Método**: {pred['metodo']}
            - **Base (mediana)**: {pred['base_sin_ajustar']}h
            - **Después de ajustes**: {pred['horas_centrales']}h
            - **Casos históricos**: {pred['casos_historicos']}
            """)
        
        with col_det2:
            st.markdown("### 🔍 Factores Considerados")
            if pred['factores']:
                for factor in pred['factores']:
                    st.markdown(f"• {factor}")
            else:
                st.info("No se aplicaron ajustes adicionales")
        
        # Recomendaciones
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones")
        
        if pred['confianza'] == 'Alta':
            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>Estimación confiable</strong> basada en {pred['casos_historicos']} casos históricos similares del cliente.
            </div>
            """, unsafe_allow_html=True)
        elif pred['confianza'] == 'Media':
            st.markdown("""
            <div class="warning-box">
            ⚠️ <strong>Estimación moderada</strong> - Considerar revisión con el equipo antes de comprometer.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            ❌ <strong>Baja confianza</strong> - Cliente sin historial suficiente.<br>
            Considerar spike técnico de 2-4h para análisis inicial.
            </div>
            """, unsafe_allow_html=True)
        
        if pred['horas_centrales'] > 12:
            st.markdown("""
            <div class="warning-box">
            🚨 <strong>Bug complejo detectado</strong> - Considerar:<br>
            • División en subtareas<br>
            • Revisión técnica con senior<br>
            • Validación de accesos/permisos con cliente
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR: INFO Y ESTADÍSTICAS
# ============================================================

with st.sidebar:
    st.header("ℹ️ Información")
    
    st.markdown("""
    ### Cómo funciona
    
    El sistema estima el esfuerzo usando:
    
    1. **Base estadística**: Mediana histórica del cliente
    2. **Reglas de negocio**: Ajustes por keywords (IRPF, licencias, masivo, etc.)
    3. **Intervalos**: MAE específico por cliente
    
    ### Precisión
    
    - MAE: ~3.5h
    - 65% predicciones dentro de ±5h
    - Basado en 128 casos históricos
    
    ### Clientes
    """)
    
    # Mostrar stats por cliente
    for cli, stats in estimador.stats_cliente.items():
        if stats:
            st.markdown(f"""
            **{cli}**  
            Mediana: {stats['median']:.1f}h | Casos: {stats['count']}
            """)
    
    st.markdown("---")
    st.markdown("""
    <small>
    Versión 1.0<br>
    Última actualización: 2025-01-20<br>
    Autor: Diego Flores
    </small>
    """, unsafe_allow_html=True)
