"""
model_deploy.py

Disponibilizacion del modelo de riesgo crediticio.

Combina el modelo heuristico de reglas duras con el modelo estadistico en una
unica decision. Las reglas se evaluan primero y prevalecen: cuando alguna se
activa, la solicitud se rechaza con un motivo explicito, sin importar la
probabilidad que estime el modelo. Solo las solicitudes que ninguna regla
marca llegan al clasificador.

    solicitud
        |
    limpieza y variables derivadas
        |
    reglas duras --- activa ---> RECHAZO (motivo auditable)
        |
    ninguna activa
        |
    modelo --> probabilidad --> umbral --> APROBADO / RECHAZADO
"""

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (  # noqa: E402
    cargar_config,
    construir_pipeline_limpieza,
)
from hueristic_model import construir_modelo  # noqa: E402

logger = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_MODELO = RAIZ_PROYECTO / "mlops_pipeline" / "modelo_final.joblib"

APROBADO = "APROBADO"
RECHAZADO_POR_REGLA = "RECHAZADO_POR_REGLA"
RECHAZADO_POR_MODELO = "RECHAZADO_POR_MODELO"
FUERA_DE_ALCANCE = "FUERA_DE_ALCANCE"


def cargar_artefacto(ruta: Path = RUTA_MODELO) -> dict[str, Any]:
    """Carga el pipeline entrenado junto con sus metadatos.

    Raises:
        FileNotFoundError: si el modelo no ha sido generado todavia.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {ruta}. "
            "Ejecute primero model_training_evaluation.py."
        )
    artefacto = joblib.load(ruta)
    logger.info(
        "Modelo cargado: %s (%s)",
        artefacto["metadatos"]["modelo"],
        artefacto["metadatos"]["metodo_optimizacion"],
    )
    return artefacto


_pipeline_limpieza = None


def obtener_pipeline_limpieza(config: dict[str, Any],
                              artefacto: dict[str, Any] | None = None):
    """Devuelve el pipeline de limpieza ajustado sobre el dataset historico.

    Se toma del artefacto guardado durante el entrenamiento. Asi el despliegue
    no necesita acceso al dataset original, lo que ademas evita reajustar el
    pipeline con cada lote: decisiones aprendidas, como que columnas superan
    el umbral de nulos, dependerian entonces del lote recibido.
    """
    global _pipeline_limpieza
    if _pipeline_limpieza is None:
        artefacto = artefacto or cargar_artefacto()
        _pipeline_limpieza = artefacto["metadatos"]["pipeline_limpieza"]
        logger.info("Pipeline de limpieza recuperado del artefacto")
    return _pipeline_limpieza


def preparar_solicitudes(crudas: pd.DataFrame,
                         config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Index]:
    """Aplica limpieza y variables derivadas a solicitudes nuevas.

    Solo transforma: los parametros aprendidos provienen del dataset
    historico. El pipeline elimina filas fuera del alcance del modelo, como
    los tipos de credito no contemplados; en produccion una solicitud no
    puede desaparecer sin respuesta, asi que se comparan los indices de
    entrada y salida y las descartadas se reportan aparte.

    Returns:
        Tupla con las solicitudes preparadas y los indices de las que
        quedaron fuera del alcance.
    """
    entrada = crudas.copy()
    entrada.index = pd.RangeIndex(len(entrada))

    objetivo = config["datos"]["columna_objetivo"]
    if objetivo in entrada.columns:
        entrada = entrada.drop(columns=[objetivo])

    pipeline = obtener_pipeline_limpieza(config)
    preparadas = entrada
    for _, paso in pipeline.steps:
        preparadas = paso.transform(preparadas)

    fuera_de_alcance = entrada.index.difference(preparadas.index)

    if len(fuera_de_alcance):
        logger.warning(
            "%d solicitudes quedaron fuera del alcance del modelo",
            len(fuera_de_alcance),
        )

    return preparadas, fuera_de_alcance


def decidir(crudas: pd.DataFrame, config: dict[str, Any],
            artefacto: dict[str, Any] | None = None) -> pd.DataFrame:
    """Evalua un lote de solicitudes y devuelve la decision de cada una.

    Args:
        crudas: solicitudes con las columnas del dataset original.
        config: configuracion del proyecto.
        artefacto: modelo ya cargado. Si se omite, se carga del disco.

    Returns:
        DataFrame con una fila por solicitud, indicando decision, motivo y
        probabilidad estimada de incumplimiento.
    """
    artefacto = artefacto or cargar_artefacto()
    pipeline = artefacto["pipeline"]
    umbral = config["decision"]["umbral_corte"]

    preparadas, fuera_de_alcance = preparar_solicitudes(crudas, config)

    resultado = pd.DataFrame(
        index=pd.RangeIndex(len(crudas)),
        columns=["decision", "motivo", "probabilidad_incumplimiento"],
    )

    resultado.loc[fuera_de_alcance, "decision"] = FUERA_DE_ALCANCE
    resultado.loc[fuera_de_alcance, "motivo"] = (
        "Tipo de credito fuera del alcance declarado del modelo"
    )

    if preparadas.empty:
        return resultado

    # 1. Reglas duras: prevalecen sobre la prediccion estadistica.
    heuristico = construir_modelo(config)
    heuristico.fit(preparadas)
    explicacion = heuristico.explicar(preparadas)
    rechazados_por_regla = explicacion["estado"] == "RECHAZO"

    resultado.loc[explicacion.index[rechazados_por_regla], "decision"] = (
        RECHAZADO_POR_REGLA
    )
    resultado.loc[explicacion.index[rechazados_por_regla], "motivo"] = (
        explicacion.loc[rechazados_por_regla, "motivos"].values
    )

    # 2. Modelo estadistico, solo sobre lo que ninguna regla marco.
    indeterminados = explicacion.index[~rechazados_por_regla]
    if len(indeterminados):
        probabilidades = pipeline.predict_proba(
            preparadas.loc[indeterminados]
        )[:, 1]
        resultado.loc[indeterminados, "probabilidad_incumplimiento"] = (
            probabilidades.round(4)
        )

        supera_umbral = probabilidades >= umbral
        resultado.loc[indeterminados[supera_umbral], "decision"] = (
            RECHAZADO_POR_MODELO
        )
        resultado.loc[indeterminados[supera_umbral], "motivo"] = (
            f"Probabilidad de incumplimiento igual o superior a {umbral}"
        )
        resultado.loc[indeterminados[~supera_umbral], "decision"] = APROBADO
        resultado.loc[indeterminados[~supera_umbral], "motivo"] = (
            "No se activaron reglas duras y el riesgo estimado esta bajo el umbral"
        )

    return resultado


def resumir_decisiones(decisiones: pd.DataFrame) -> pd.DataFrame:
    """Agrupa el resultado del lote por tipo de decision."""
    resumen = (
        decisiones["decision"]
        .value_counts()
        .rename_axis("decision")
        .reset_index(name="solicitudes")
    )
    resumen["porcentaje"] = (
        resumen["solicitudes"] / len(decisiones) * 100
    ).round(2)
    return resumen


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = cargar_config()
    muestra = pd.read_csv(
        RAIZ_PROYECTO / config["datos"]["ruta"],
        sep=config["datos"]["separador"],
    ).head(200)

    decisiones = decidir(muestra, config)
    logger.info("\n%s", resumir_decisiones(decisiones).to_string(index=False))
    logger.info("Ejemplo de decisiones:\n%s", decisiones.head(10).to_string())