"""
model_monitoring.py

Monitoreo del modelo en produccion y deteccion de datadrift.

En riesgo crediticio el resultado real de una solicitud se conoce meses
despues del desembolso, cuando la obligacion vence o se paga. No es posible,
por tanto, medir el acierto del modelo de forma continua. El monitoreo se
concentra en lo que si es observable de inmediato:

    1. Que la poblacion que llega al modelo siga pareciendose a la que uso
       para entrenar (deriva de las variables de entrada).
    2. Que la distribucion de las decisiones se mantenga estable (deriva de
       las predicciones).

Una deriva sostenida indica que el modelo esta prediciendo sobre una
poblacion distinta de la que aprendio, y que sus estimaciones han dejado de
ser confiables aunque nada falle de forma visible.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import cargar_config  # noqa: E402
from model_deploy import RAIZ_PROYECTO, preparar_solicitudes  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# REGISTRO DE PREDICCIONES
# ============================================================================
def registrar_predicciones(entradas: pd.DataFrame, decisiones: pd.DataFrame,
                           config: dict[str, Any]) -> Path:
    """Acumula las solicitudes evaluadas junto con la decision entregada.

    Sin este registro no hay monitoreo posible: la deriva se mide comparando
    lo que llega hoy contra lo que llegaba cuando se entreno el modelo, y eso
    exige conservar el historial de lo que paso por el endpoint.

    Args:
        entradas: solicitudes tal como se recibieron.
        decisiones: salida de model_deploy.decidir.
        config: configuracion del proyecto.

    Returns:
        Ruta del archivo de registro.
    """
    ruta = RAIZ_PROYECTO / config["monitoreo"]["ruta_registro"]
    ruta.parent.mkdir(parents=True, exist_ok=True)

    lote = pd.concat(
        [entradas.reset_index(drop=True), decisiones.reset_index(drop=True)],
        axis=1,
    )
    lote.insert(0, "fecha_evaluacion", datetime.now(timezone.utc).isoformat())

    lote.to_csv(
        ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8"
    )
    logger.info("Registradas %d predicciones en %s", len(lote), ruta.name)
    return ruta


def cargar_registro(config: dict[str, Any]) -> pd.DataFrame:
    """Lee el historial acumulado de predicciones.

    Raises:
        FileNotFoundError: si todavia no se ha evaluado ninguna solicitud.
    """
    ruta = RAIZ_PROYECTO / config["monitoreo"]["ruta_registro"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No hay registro de predicciones en {ruta}. "
            "El monitoreo requiere solicitudes evaluadas previamente."
        )
    registro = pd.read_csv(ruta)
    registro["fecha_evaluacion"] = pd.to_datetime(registro["fecha_evaluacion"])
    logger.info("Registro cargado: %d predicciones", len(registro))
    return registro


# ============================================================================
# MEDIDAS DE DERIVA
# ============================================================================
def indice_estabilidad(referencia: pd.Series, actual: pd.Series,
                       bins: int = 10) -> float:
    """Calcula el Population Stability Index entre dos distribuciones.

    El PSI es la medida estandar de deriva en riesgo crediticio. Divide la
    distribucion de referencia en tramos de igual frecuencia y compara que
    proporcion de la poblacion actual cae en cada uno. Valores habituales de
    interpretacion en la industria:

        menor a 0,10   poblacion estable
        0,10 a 0,25    deriva moderada, conviene revisar
        mayor a 0,25   deriva significativa, el modelo requiere reentrenamiento

    Args:
        referencia: distribucion con la que se entreno el modelo.
        actual: distribucion observada en produccion.
        bins: numero de tramos.

    Returns:
        Valor del indice. Cuanto mayor, mas se alejo la poblacion.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()
    if len(referencia) < bins or len(actual) == 0:
        return np.nan

    cortes = np.unique(
        np.quantile(referencia, np.linspace(0, 1, bins + 1))
    )
    if len(cortes) < 3:
        return np.nan
    cortes[0], cortes[-1] = -np.inf, np.inf

    prop_ref = np.histogram(referencia, bins=cortes)[0] / len(referencia)
    prop_act = np.histogram(actual, bins=cortes)[0] / len(actual)

    # Se sustituyen las proporciones nulas por un valor minimo para evitar
    # divisiones por cero y logaritmos indefinidos.
    minimo = 1e-6
    prop_ref = np.where(prop_ref == 0, minimo, prop_ref)
    prop_act = np.where(prop_act == 0, minimo, prop_act)

    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))


def _deriva_categorica(referencia: pd.Series, actual: pd.Series) -> float:
    """Mide la deriva de una variable categorica con la V de Cramer.

    Queda en el rango [0, 1] y es comparable entre variables con distinto
    numero de categorias.
    """
    tabla = pd.crosstab(
        pd.concat([referencia, actual]),
        ["referencia"] * len(referencia) + ["actual"] * len(actual),
    )
    if min(tabla.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tabla)[0]
    n = tabla.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))


def evaluar_deriva(referencia: pd.DataFrame, actual: pd.DataFrame,
                   config: dict[str, Any]) -> pd.DataFrame:
    """Compara variable por variable la poblacion actual contra la historica.

    Para las numericas se reportan dos medidas complementarias: el PSI, que
    cuantifica la magnitud del desplazamiento, y la prueba de
    Kolmogorov-Smirnov, que indica si la diferencia es estadisticamente
    detectable. Una muestra grande puede volver significativa una diferencia
    irrelevante, asi que el PSI es el criterio de decision y el valor p es
    informacion complementaria.

    Returns:
        DataFrame con una fila por variable, ordenado por severidad.
    """
    parametros = config["monitoreo"]
    comunes = [c for c in referencia.columns if c in actual.columns]
    filas = []

    for variable in comunes:
        serie_ref = referencia[variable]
        serie_act = actual[variable]

        es_discreta = serie_ref.dropna().nunique() <= 10
        if pd.api.types.is_numeric_dtype(serie_ref) and not es_discreta:
            psi = indice_estabilidad(serie_ref, serie_act, parametros["bins_psi"])
            valida = serie_ref.notna().sum() > 0 and serie_act.notna().sum() > 0
            p_valor = (
                ks_2samp(serie_ref.dropna(), serie_act.dropna()).pvalue
                if valida else np.nan
            )
            medida, tipo = psi, "numerica"
        else:
            medida = _deriva_categorica(
                serie_ref.astype(str), serie_act.astype(str))
            p_valor, tipo = np.nan, "categorica"

        if pd.isna(medida):
            estado = "sin_datos"
        elif medida >= parametros["umbral_psi_critico"]:
            estado = "CRITICO"
        elif medida >= parametros["umbral_psi_alerta"]:
            estado = "alerta"
        else:
            estado = "estable"

        filas.append({
            "variable": variable,
            "tipo": tipo,
            "medida_deriva": round(medida, 4) if pd.notna(medida) else None,
            "ks_pvalor": round(p_valor, 6) if pd.notna(p_valor) else None,
            "estado": estado,
        })

    resultado = pd.DataFrame(filas).sort_values(
        "medida_deriva", ascending=False, na_position="last"
    )

    criticas = (resultado["estado"] == "CRITICO").sum()
    alertas = (resultado["estado"] == "alerta").sum()
    logger.info(
        "Deriva de variables: %d criticas, %d en alerta, %d estables",
        criticas, alertas, (resultado["estado"] == "estable").sum(),
    )
    return resultado

# ============================================================================
# DERIVA DE LAS PREDICCIONES
# ============================================================================
def evaluar_deriva_decisiones(registro: pd.DataFrame,
                              config: dict[str, Any]) -> dict[str, Any]:
    """Vigila que la distribucion de decisiones se mantenga estable.

    Es el indicador mas temprano de que algo cambio. Las variables de entrada
    pueden moverse poco individualmente y aun asi alterar el resultado
    conjunto, porque el modelo combina todas a la vez. Un salto en la tasa de
    rechazo exige explicacion aunque ninguna variable haya derivado por si
    sola.

    Returns:
        Diccionario con la tasa observada, la esperada y el estado.
    """
    parametros = config["monitoreo"]
    esperada = parametros["tasa_rechazo_esperada"]
    tolerancia = parametros["tolerancia_tasa_rechazo"]

    rechazos = registro["decision"].astype(str).str.startswith("RECHAZADO")
    observada = float(rechazos.mean())
    desviacion = observada - esperada

    estado = "estable" if abs(desviacion) <= tolerancia else "ALERTA"

    logger.info("--- Deriva de decisiones ---")
    logger.info("    Tasa de rechazo observada: %.4f", observada)
    logger.info("    Tasa esperada:             %.4f (tolerancia +/- %.2f)",
                esperada, tolerancia)
    logger.info("    Desviacion:                %+.4f -> %s", desviacion, estado)

    reparto = registro["decision"].value_counts(normalize=True).round(4)
    logger.info("    Reparto de decisiones:\n%s", reparto.to_string())

    probabilidades = pd.to_numeric(
        registro["probabilidad_incumplimiento"], errors="coerce"
    ).dropna()
    resumen_prob = {}
    if not probabilidades.empty:
        resumen_prob = {
            "media": round(probabilidades.mean(), 4),
            "mediana": round(probabilidades.median(), 4),
            "p90": round(probabilidades.quantile(0.90), 4),
        }
        logger.info("    Probabilidades estimadas: %s", resumen_prob)

    return {
        "tasa_observada": observada,
        "tasa_esperada": esperada,
        "desviacion": desviacion,
        "estado": estado,
        "reparto": reparto.to_dict(),
        "probabilidades": resumen_prob,
    }


# ============================================================================
# REPORTE
# ============================================================================
def ejecutar_monitoreo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta el ciclo completo de monitoreo y produce el reporte.

    Compara la poblacion registrada en produccion contra el dataset con el
    que se entreno el modelo, y evalua tanto las variables de entrada como la
    distribucion de las decisiones.

    Returns:
        Diccionario con la tabla de deriva por variable, el diagnostico de
        decisiones y la conclusion general.
    """
    config = config or cargar_config()

    historico = pd.read_csv(
        RAIZ_PROYECTO / config["datos"]["ruta"],
        sep=config["datos"]["separador"],
    )
    referencia, _ = preparar_solicitudes(historico, config)

    registro = cargar_registro(config)
    columnas_decision = {
        "fecha_evaluacion", "decision", "motivo", "probabilidad_incumplimiento",
    }
    entradas = registro.drop(columns=columnas_decision, errors="ignore")
    actual, _ = preparar_solicitudes(entradas, config)

    logger.info(
        "Referencia: %d registros | Produccion: %d registros",
        len(referencia), len(actual),
    )

    deriva_variables = evaluar_deriva(referencia, actual, config)
    logger.info("--- Deriva por variable ---")
    logger.info("\n%s", deriva_variables.to_string(index=False))

    deriva_decisiones = evaluar_deriva_decisiones(registro, config)

    criticas = deriva_variables.loc[
        deriva_variables["estado"] == "CRITICO", "variable"
    ].tolist()

    if criticas:
        conclusion = (
            f"Reentrenamiento recomendado. Variables con deriva significativa: "
            f"{', '.join(criticas)}."
        )
    elif deriva_decisiones["estado"] == "ALERTA":
        conclusion = (
            "Las variables individuales se mantienen estables, pero la tasa de "
            "rechazo se desvio de lo esperado. Revisar el efecto conjunto."
        )
    elif (deriva_variables["estado"] == "alerta").any():
        conclusion = (
            "Deriva moderada en algunas variables. Mantener seguimiento sin "
            "accion inmediata."
        )
    else:
        conclusion = "Poblacion estable. El modelo opera sobre datos comparables."

    logger.info("--- Conclusion ---")
    logger.info("    %s", conclusion)

    ruta_reporte = (
        RAIZ_PROYECTO / "mlops_pipeline" / "reportes" / "reporte_datadrift.csv"
    )
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    deriva_variables.to_csv(ruta_reporte, index=False)
    logger.info("Reporte guardado en %s", ruta_reporte)

    return {
        "deriva_variables": deriva_variables,
        "deriva_decisiones": deriva_decisiones,
        "conclusion": conclusion,
    }


# ============================================================================
# SIMULACION
# ============================================================================
def simular_poblacion_derivada(historico: pd.DataFrame,
                               n: int = 500,
                               semilla: int = 42) -> pd.DataFrame:
    """Genera un lote con deriva inducida para verificar el monitoreo.

    Sin datos futuros reales no hay forma de comprobar que el sistema detecta
    un cambio poblacional. Esta funcion desplaza deliberadamente algunas
    variables para confirmar que las alertas se disparan: un monitoreo que
    nunca se ha visto reaccionar no ofrece garantia alguna.

    El desplazamiento simula un escenario plausible de relajacion de los
    criterios de originacion: clientes mas jovenes, de menores ingresos y con
    peor comportamiento en centrales.
    """
    generador = np.random.default_rng(semilla)
    muestra = historico.sample(n=n, random_state=semilla).copy()

    muestra["edad_cliente"] = (
        muestra["edad_cliente"] * generador.uniform(0.65, 0.80, n)
    ).round()
    muestra["salario_cliente"] = (
        muestra["salario_cliente"] * generador.uniform(0.50, 0.70, n)
    ).round()
    muestra["puntaje_datacredito"] = (
        muestra["puntaje_datacredito"] * generador.uniform(0.70, 0.85, n)
    ).clip(150, 950).round()
    muestra["huella_consulta"] = (
        muestra["huella_consulta"] + generador.integers(2, 8, n)
    )

    return muestra.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from model_deploy import decidir

    config = cargar_config()
    historico = pd.read_csv(
        RAIZ_PROYECTO / config["datos"]["ruta"],
        sep=config["datos"]["separador"],
    )

    registro = RAIZ_PROYECTO / config["monitoreo"]["ruta_registro"]
    if registro.exists():
        registro.unlink()
        logger.info("Registro previo eliminado (modo demostracion)")

    logger.info("=== Demostracion · Lote 1: poblacion comparable ===")
    lote_normal = historico.sample(n=500, random_state=7).reset_index(drop=True)
    registrar_predicciones(lote_normal, decidir(lote_normal, config), config)
    ejecutar_monitoreo(config)

    logger.info("=== Lote 2: poblacion con deriva inducida ===")
    lote_derivado = simular_poblacion_derivada(historico, n=500)
    registrar_predicciones(lote_derivado, decidir(lote_derivado, config), config)
    ejecutar_monitoreo(config)