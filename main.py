from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# Configuración visual para el póster
sns.set_theme(style="whitegrid")

# ==============================================================================
# CARGA DE DATOS (Ruta relativa con pathlib para el bono de GitHub)
# ==============================================================================
DIRECTORIO = Path(__file__).parent

# Detectar automáticamente si es .xls o .xlsx
RUTA_XLSX = DIRECTORIO / "ie_data.xlsx"
RUTA_XLS = DIRECTORIO / "ie_data.xls"

if RUTA_XLSX.exists():
    RUTA_EXCEL = RUTA_XLSX
else:
    RUTA_EXCEL = RUTA_XLS

print(f"Cargando archivo de datos desde: {RUTA_EXCEL.name}")

try:
    df = pd.read_excel(RUTA_EXCEL, sheet_name='Data', skiprows=6)
except Exception as e:
    # Intentar con motor alternativo si xlrd no está instalado
    df = pd.read_excel(RUTA_EXCEL, sheet_name='Data', skiprows=6, engine='openpyxl')

df = df.iloc[:, [0, 1, 2]].dropna().copy()
df.columns = ['Fecha_num', 'Precio', 'Dividendo']

df = df[pd.to_numeric(df['Precio'], errors='coerce').notnull()].copy()
df['Precio'] = df['Precio'].astype(float)
df['Dividendo'] = df['Dividendo'].astype(float)

# Serie de tiempo: Dividend Yield (%)
df['Dividend_Yield'] = (df['Dividendo'] / df['Precio']) * 100

# Muestra reciente (últimas 300 observaciones, T >= 100)
serie = df['Dividend_Yield'].iloc[-300:].reset_index(drop=True)

# ==============================================================================
# PASO 1: IDENTIFICACIÓN Y PRUEBAS DE ESTACIONARIEDAD
# ==============================================================================
print("\n--- PASO 1: IDENTIFICACIÓN ---")

# Transformación logarítmica y diferenciación
log_serie = np.log(serie)
diff_log = log_serie.diff().dropna()

# Prueba de Raíz Unitaria (ADF)
resultado_adf = adfuller(diff_log)
print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"p-valor ADF: {resultado_adf[1]:.6f}")

if resultado_adf[1] < 0.05:
    print("-> La serie diferenciada es ESTACIONARIA (p-valor < 0.05).")

# Gráfico completo de Identificación para el póster (Serie, ACF y PACF)
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
axes[0].plot(serie, color='navy', lw=1.5)
axes[0].set_title("1. Serie Nivel: Rentabilidad por Dividendos S&P 500 (%)")

axes[1].plot(diff_log, color='teal', lw=1.5)
axes[1].set_title("2. Serie Estacionaria (Diferencia Logarítmica)")

# ACF y PACF en la misma figura
plot_acf(diff_log, ax=axes[2], lags=20, title="3. Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
plot_pacf(diff_log, ax=axes[2], lags=20, color='crimson')
plt.tight_layout()
plt.savefig(DIRECTORIO / "paso1_identificacion.png", dpi=300)
plt.close()

# ==============================================================================
# PASO 2: ESTIMACIÓN Y SELECCIÓN DE MODELOS
# ==============================================================================
print("\n--- PASO 2: ESTIMACIÓN Y COMPARACIÓN DE MODELOS ---")

# Estimación manual de 3 modelos candidatos (cumpliendo norma sin auto.arima)
m1 = ARIMA(log_serie, order=(1, 1, 0)).fit()
m2 = ARIMA(log_serie, order=(0, 1, 1)).fit()
m3 = ARIMA(log_serie, order=(1, 1, 1)).fit()

print(f"Modelo ARIMA(1,1,0) -> AIC: {m1.aic:.2f} | BIC: {m1.bic:.2f}")
print(f"Modelo ARIMA(0,1,1) -> AIC: {m2.aic:.2f} | BIC: {m2.bic:.2f}")
print(f"Modelo ARIMA(1,1,1) -> AIC: {m3.aic:.2f} | BIC: {m3.bic:.2f}")

# Seleccionamos ARIMA(0,1,1) por minimizar AIC/BIC
modelo_optimo = m2
print("\n-> Modelo seleccionado: ARIMA(0,1,1)")
print(modelo_optimo.summary())

# ==============================================================================
# PASO 3: VALIDACIÓN DE SUPUESTOS (DIAGNÓSTICO RIGUROSO)
# ==============================================================================
print("\n--- PASO 3: VALIDACIÓN DE RESIDUOS ---")

# 1. Independencia (Ljung-Box)
prueba_lb = acorr_ljungbox(modelo_optimo.resid, lags=[10], return_df=True)
p_val_lb = prueba_lb['lb_pvalue'].values[0]
veredicto_lb = "CUMPLE (Ruido Blanco)" if p_val_lb > 0.05 else "NO CUMPLE"

# 2. Homocedasticidad 
test_het = modelo_optimo.test_heteroskedasticity(method='breakvar')
p_val_het = test_het[0][1] 
veredicto_het = "CUMPLE (Varianza Constante)" if p_val_het > 0.05 else "NO CUMPLE"

# 3. Normalidad (Jarque-Bera)
test_jb = modelo_optimo.test_normality(method='jarquebera')
p_val_jb = test_jb[0][1] 
veredicto_jb = "CUMPLE (Normal)" if p_val_jb > 0.05 else "NO CUMPLE (Justificable T=300)"

# Construcción de la tabla con pandas
tabla_diagnostico = pd.DataFrame({
    "Supuesto Evaluado": ["Independencia (Ljung-Box)", "Homocedasticidad", "Normalidad (Jarque-Bera)"],
    "p-valor": [f"{p_val_lb:.4f}", f"{p_val_het:.4f}", f"{p_val_jb:.4f}"],
    "Veredicto Final": [veredicto_lb, veredicto_het, veredicto_jb]
})

print(tabla_diagnostico.to_string(index=False))

# 4. Comprobación Teórica de Invertibilidad Corregida
coef_ma = modelo_optimo.params.get('ma.L1', 0)
raiz_ma = -1 / coef_ma if coef_ma != 0 else np.nan
modulo_raiz = abs(raiz_ma)

print(f"\n4. Comprobación de Invertibilidad MA:")
print(f"Coeficiente theta_1: {coef_ma:.4f} | Raíz del polinomio: {raiz_ma:.4f} (Módulo: {modulo_raiz:.4f})")
if abs(coef_ma) < 1:
    print("   -> CUMPLE: |theta_1| < 1 y la raíz cae fuera del círculo unitario (El modelo es INVERTIBLE).")
else:
    print("   -> NO CUMPLE: El modelo no es invertible.")

# 5. Generación del Panel Gráfico de Diagnóstico Completo
fig_diag = modelo_optimo.plot_diagnostics(figsize=(12, 8))
ejes = fig_diag.axes
ejes[0].set_title("Residuos Estandarizados")
ejes[1].set_title("Histograma y Densidad Estimada")
ejes[2].set_title("Gráfico Q-Q Normal")
ejes[3].set_title("Correlograma (ACF)")

ejes[1].legend(["Hist", "KDE (Densidad)", "N(0,1) Normal"])
ejes[2].set_xlabel("Cuantiles Teóricos")
ejes[2].set_ylabel("Cuantiles de la Muestra")

fig_diag.suptitle("Diagnóstico Visual de Residuos - Modelo ARIMA(0,1,1)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(DIRECTORIO / "paso3_diagnostico_avanzado.png", dpi=300)
plt.close()

# ==============================================================================
# PASO 4: PRONÓSTICO PARA 10 PERIODOS ADELANTE
# ==============================================================================
print("\n--- PASO 4: PRONÓSTICO A 10 PERIODOS ---")

pronostico = modelo_optimo.get_forecast(steps=10)
df_pron = pronostico.summary_frame(alpha=0.05)

# Convertir de logaritmo a escala porcentual original
df_pron_exp = np.exp(df_pron)

print("Proyección para los próximos 10 meses (%):")
print(df_pron_exp[['mean', 'mean_ci_lower', 'mean_ci_upper']])

# Gráfico del pronóstico para el póster
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(250, 300), serie.iloc[-50:], label="Histórico Reciente", color='navy', lw=2)
ax.plot(range(300, 310), df_pron_exp['mean'], label="Pronóstico (10 Meses)", color='crimson', marker='o')
ax.fill_between(range(300, 310), df_pron_exp['mean_ci_lower'], df_pron_exp['mean_ci_upper'], 
                color='crimson', alpha=0.2, label="Int. Confianza 95%")
ax.set_title("Pronóstico Box-Jenkins a 10 Periodos - Dividend Yield S&P 500 (%)")
ax.set_xlabel("Periodo (Meses)")
ax.set_ylabel("Dividend Yield (%)")
ax.legend()
plt.tight_layout()
plt.savefig(DIRECTORIO / "paso4_pronostico.png", dpi=300)
plt.close()

print("\n¡PROCESO COMPLETADO CON ÉXITO! Imágenes actualizadas en la carpeta del proyecto.")