"""
Estimador de Esfuerzo - Payroll
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
import json
from scipy import stats
from typing import Dict

# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================

st.set_page_config(
    page_title="Estimador Payroll",
    page_icon="🕒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
def load_css():
    css_path = Path(__file__).with_name("index.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )

load_css()

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


class EstimadorDiez:
    """Estimador para Payroll Diez basado en config JSON."""

    def __init__(self, config: Dict):
        self.config = config
        self.global_mediana = float(config.get("global", {}).get("mediana", 0.0))
        self.global_mad = float(config.get("global", {}).get("mad", 0.0))
        self.keywords = config.get("keywords", {})
        self.clientes = config.get("clientes", {})
        self.n_bugs = int(config.get("n_bugs", 0))

    def _aplicar_keywords(self, texto: str) -> tuple:
        texto_lower = texto.lower()
        multiplicador = 1.0
        factores = []

        for keyword, factor in self.keywords.items():
            try:
                factor_val = float(factor)
            except (TypeError, ValueError):
                continue
            if keyword.lower() in texto_lower:
                multiplicador *= factor_val
                pct = int(round((factor_val - 1) * 100))
                nombre = str(keyword).replace("_", " ").title()
                if pct != 0:
                    factores.append(f"{nombre} (+{pct}%)")
                else:
                    factores.append(nombre)

        return multiplicador, factores

    def predecir(self, cliente: str, summary: str, description: str = "") -> Dict:
        stats_cliente = self.clientes.get(cliente)

        if stats_cliente:
            base = float(stats_cliente.get("med", self.global_mediana))
            mae = float(stats_cliente.get("mad", self.global_mad))
            confianza = str(stats_cliente.get("conf", "Media"))
            casos = int(stats_cliente.get("n", 0))
            metodo = f"Mediana cliente ({casos} casos)" if casos else "Mediana cliente"
        else:
            base = self.global_mediana
            mae = self.global_mad
            confianza = "Baja"
            casos = 0
            metodo = "Mediana global (sin historial)"

        texto_completo = f"{summary} {description}"
        multiplicador, factores = self._aplicar_keywords(texto_completo)

        pred_central = base * multiplicador
        pred_min = max(0.5, pred_central - mae)
        pred_max = pred_central + mae

        pred_central = round(pred_central * 4) / 4
        pred_min = round(pred_min * 4) / 4
        pred_max = round(pred_max * 4) / 4

        if confianza.lower() not in ("alta", "media"):
            factores.append("Cliente sin historial - rango amplio")

        return {
            'horas_centrales': pred_central,
            'intervalo_min': pred_min,
            'intervalo_max': pred_max,
            'confianza': confianza,
            'factores': factores,
            'metodo': metodo,
            'casos_historicos': casos,
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

estimador_diecisiete = cargar_estimador()

@st.cache_resource
def cargar_estimador_diez():
    """Carga el estimador de Payroll Diez desde config JSON."""
    try:
        config_candidates = [
            Path("modelo_diez_config.json"),
            Path("modelo_diez_config (2).json"),
        ]
        config_path = next((path for path in config_candidates if path.exists()), None)
        if not config_path:
            st.error("Archivo modelo_diez_config.json no encontrado")
            st.stop()
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return EstimadorDiez(config)
    except Exception as e:
        st.error(f"Error al cargar modelo Diez: {str(e)}")
        st.stop()

estimador_diez = cargar_estimador_diez()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="brand-bar">
    <div class="brand-logo">PAYROLL</div>
</div>
<div class="page-title">
    <h1>Estimador de Esfuerzo - Payroll</h1>
    <p>Sistema basado en estadísticas robustas + reglas de negocio calibradas</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FORMULARIO
# ============================================================

estimador_activo = estimador_diecisiete

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="card card-blue">
        <div class="card-title">
            <span class="icon icon-blue">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M8 3h6l4 4v14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>
                    <path d="M14 3v5h5"/>
                    <path d="M9 13h6M9 17h6"/>
                </svg>
            </span>
            Información del Bug
        </div>
    """, unsafe_allow_html=True)

    producto = st.selectbox(
        "Producto *",
        options=["Payroll Diecisiete", "Payroll Diez"],
        help="Selecciona el producto"
    )

    if producto == "Payroll Diez":
        estimador_activo = estimador_diez
        clientes = sorted(estimador_diez.clientes.keys())
    else:
        estimador_activo = estimador_diecisiete
        clientes = sorted(estimador_diecisiete.stats_cliente.keys())

    # Cliente
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

    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="card card-teal">
        <div class="card-title">
            <span class="icon icon-teal">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 4h16v16H4z"/>
                    <path d="M8 8h8M8 12h8M8 16h6"/>
                </svg>
            </span>
            Descripción Detallada
        </div>
    """, unsafe_allow_html=True)

    descripcion = st.text_area(
        "Descripción (opcional)",
        placeholder="Detallar pasos, contexto y comportamiento observado"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Boton estimar
st.markdown('<hr class="form-divider">', unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

with col_btn1:
    btn_estimate = st.button("Estimar Esfuerzo", type="primary", use_container_width=True)

with col_btn2:
    btn_clear = st.button("Limpiar", use_container_width=True)
# ============================================================
# LOGICA DE ESTIMACION
# ============================================================

if btn_clear:
    st.rerun()

if btn_estimate:
    if not summary:
        st.warning("Por favor ingresa un resumen del bug")
    else:
        # Predecir
        pred = estimador_activo.predecir(
            cliente=cliente,
            summary=summary,
            description=descripcion
        )

        confidence_class = "green" if pred['confianza'] == "Alta" else "blue" if pred['confianza'] == "Media" else "navy"

        st.markdown(f"""
        <div class="card card-result">
            <div class="card-title">
                <span class="icon icon-blue">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                        <path d="M4 19h16"/>
                        <path d="M6 17V9"/>
                        <path d="M12 17V5"/>
                        <path d="M18 17v-7"/>
                    </svg>
                </span>
                Resultado de la Estimación
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        <span class="icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                                <circle cx="12" cy="12" r="9"/>
                                <path d="M12 7v5l3 3"/>
                            </svg>
                        </span>
                        Horas estimadas
                    </div>
                    <div class="metric-value">{pred['horas_centrales']}h</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">
                        <span class="icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                                <path d="M12 19V5"/>
                                <path d="M7 14l5 5 5-5"/>
                            </svg>
                        </span>
                        Rango mínimo
                    </div>
                    <div class="metric-value blue">{pred['intervalo_min']}h</div>
                </div>
                <div class="metric-card teal">
                    <div class="metric-label">
                        <span class="icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                                <path d="M12 5v14"/>
                                <path d="M7 10l5-5 5 5"/>
                            </svg>
                        </span>
                        Rango máximo
                    </div>
                    <div class="metric-value teal">{pred['intervalo_max']}h</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">
                        <span class="icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                                <circle cx="12" cy="12" r="9"/>
                                <path d="M8 12l3 3 5-5"/>
                            </svg>
                        </span>
                        Confianza
                    </div>
                    <div class="metric-value {confidence_class}">{pred['confianza']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_det1, col_det2 = st.columns(2)

        with col_det1:
            st.markdown(f"""
            <div class="card card-purple">
                <div class="card-title">
                    <span class="icon icon-purple">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                            <path d="M4 17l6-6 4 4 6-6"/>
                            <path d="M4 19h16"/>
                        </svg>
                    </span>
                    Cálculo
                </div>
                <ul class="detail-list">
                    <li><span class="bullet bullet-purple"></span><strong>Método:</strong> {pred['metodo']}</li>
                    <li><span class="bullet bullet-purple"></span><strong>Base (mediana):</strong> {pred['base_sin_ajustar']}h</li>
                    <li><span class="bullet bullet-purple"></span><strong>Después de ajustes:</strong> {pred['horas_centrales']}h</li>
                    <li><span class="bullet bullet-purple"></span><strong>Casos históricos:</strong> {pred['casos_historicos']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_det2:
            if pred['factores']:
                factores_html = "".join(
                    f'<li><span class="bullet bullet-amber"></span>{factor}</li>' for factor in pred['factores']
                )
                detalle_factores = f'<ul class="detail-list">{factores_html}</ul>'
            else:
                detalle_factores = '<div class="callout warning">No se aplicaron ajustes adicionales.</div>'

            st.markdown(f"""
            <div class="card card-amber">
                <div class="card-title">
                    <span class="icon icon-amber">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                            <circle cx="11" cy="11" r="7"/>
                            <path d="M16.5 16.5L20 20"/>
                        </svg>
                    </span>
                    Factores Considerados
                </div>
                {detalle_factores}
            </div>
            """, unsafe_allow_html=True)

        if pred['confianza'] == 'Alta':
            recomendacion_texto = f"Estimación confiable basada en {pred['casos_historicos']} casos historicos similares del cliente."
            recomendacion_clase = "success"
        elif pred['confianza'] == 'Media':
            recomendacion_texto = "Estimación moderada. Considerar revisión con el equipo antes de comprometer."
            recomendacion_clase = "warning"
        else:
            recomendacion_texto = "Baja confianza. Cliente sin historial suficiente. Considerar spike técnico de 2-4h para análisis inicial."
            recomendacion_clase = "warning"

        st.markdown(f"""
        <div class="card card-success">
            <div class="card-title">
                <span class="icon icon-green">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                        <path d="M9 18l6-6"/>
                        <path d="M12 4a7 7 0 0 0-7 7c0 3.9 3.5 8 7 9 3.5-1 7-5.1 7-9a7 7 0 0 0-7-7z"/>
                    </svg>
                </span>
                Recomendaciones
            </div>
            <div class="callout {recomendacion_clase}">
                <span class="callout-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                        <circle cx="12" cy="12" r="9"/>
                        <path d="M8 12l3 3 5-5"/>
                    </svg>
                </span>
                <div>{recomendacion_texto}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if pred['horas_centrales'] > 12:
            st.markdown("""
            <div class="card card-amber">
                <div class="callout warning">
                    <span class="callout-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                            <path d="M12 3l9 16H3z"/>
                            <path d="M12 9v4"/>
                            <path d="M12 17h.01"/>
                        </svg>
                    </span>
                    <div>
                        <strong>Bug complejo detectado</strong> - Considerar:<br>
                        Dividir en subtareas<br>
                        Revisión técnica con senior<br>
                        Validación de accesos/permisos con cliente
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    <div>Copyright &copy;2026 Estimador de Esfuerzo - Payroll</div>
    <div>update 21/01/2026, 09:58 hs</div>
</div>
""", unsafe_allow_html=True)
# SIDEBAR: INFO Y ESTADÍSTICAS
# ============================================================

with st.sidebar:
    st.header("ℹ️ Información")

    if producto == "Payroll Diez":
        precision_label = "MAD"
        precision_value = estimador_diez.global_mad
        total_casos = estimador_diez.n_bugs
        keywords_ejemplo = "recibos"
    else:
        precision_label = "MAE"
        precision_value = estimador_diecisiete.global_mae
        total_casos = len(estimador_diecisiete.df_historico)
        keywords_ejemplo = "IRPF, licencias, masivo"

    st.markdown(f"""
    <div class="sidebar-section">
      <h4>Pasos</h4>
      <ol class="sidebar-list">
        <li>Seleccionar <strong>producto</strong>.</li>
        <li>Seleccionar <strong>cliente</strong>.</li>
        <li>Completar <strong>resumen</strong> y <strong>descripción</strong>.</li>
      </ol>
      <div class="sidebar-note">Tip: un resumen claro mejora la precisión.</div>
    </div>
    <div class="sidebar-section">
      <h4>Cómo funciona</h4>
      <ul class="sidebar-list">
        <li><strong>Baseline</strong>: mediana histórica por cliente.</li>
        <li><strong>Reglas</strong>: ajustes por keywords ({keywords_ejemplo},etc).</li>
        <li><strong>Intervalos</strong>: variabilidad histórica por cliente.</li>
      </ul>
    </div>
    <div class="sidebar-section">
      <h4>Precisión</h4>
      <div class="sidebar-kpi">
        <span>{precision_label}: ~{precision_value:.1f}h</span>
        <span>Casos: {total_casos}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <small>
    Versión 1.0<br>
    Última actualización: 2025-01-20<br>
    Autor: Diego Flores
    </small>
    """, unsafe_allow_html=True)
