"""evaluate.py

Prueba de metricas para el pipeline de CI (Jenkins).

Carga el modelo entrenado (`modelo_final.joblib`), lo evalua sobre el mismo
conjunto de test 70/30 que define el entrenamiento, calcula el PR-AUC sobre el
evento de incumplimiento y lo contrasta contra el piso configurado en
`config.json` -> `validacion.pr_auc_minimo`.

- Escribe el resultado en `mlops_pipeline/reportes/metricas.json` (evidencia
  para el informe y para monitoreo).
- Termina con codigo de salida 0 si el modelo pasa el piso, y 1 si no lo pasa,
  para que Jenkins marque el build en rojo cuando el desempeno cae.

Reutiliza las funciones del propio repositorio (`cargar_conjuntos`, constante
`INCUMPLE`) para que la particion y la codificacion sean identicas a las del
entrenamiento, sin duplicar logica.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

# El modulo vive en la misma carpeta src/; se asegura que sea importable
# sin importar desde donde se invoque el script (Jenkins lo corre desde la
# raiz del workspace).
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ft_engineering import cargar_config  # noqa: E402
from model_training_evaluation import cargar_conjuntos, INCUMPLE  # noqa: E402

logger = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_MODELO = RAIZ_PROYECTO / "mlops_pipeline" / "modelo_final.joblib"
RUTA_METRICAS = RAIZ_PROYECTO / "mlops_pipeline" / "reportes" / "metricas.json"


def calcular_metricas(pipeline, X_test, y_test) -> dict[str, float]:
    """Calcula las metricas de incumplimiento sobre el conjunto de test.

    La clase positiva es el incumplimiento. El pipeline persistido predice
    1 = incumplimiento, por lo que `predict_proba(X)[:, 1]` es la probabilidad
    de incumplir.
    """
    y_binario = (y_test == INCUMPLE).astype(int)
    probabilidades = pipeline.predict_proba(X_test)[:, 1]
    predicciones = pipeline.predict(X_test)

    return {
        "pr_auc": float(average_precision_score(y_binario, probabilidades)),
        "roc_auc": float(roc_auc_score(y_binario, probabilidades)),
        "recall_incumplimiento": float(
            recall_score(y_binario, predicciones, zero_division=0)),
        "f1_incumplimiento": float(
            f1_score(y_binario, predicciones, zero_division=0)),
        "n_test": int(len(y_test)),
        "tasa_incumplimiento_test": float(y_binario.mean()),
    }


def main() -> int:
    """Evalua el modelo y aplica el gate de PR-AUC. Devuelve el codigo de salida."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = cargar_config()
    umbral = config["validacion"]["pr_auc_minimo"]

    # La prueba de CI debe ser autocontenida y reproducible: se evalua sobre el
    # CSV versionado en el repositorio, no sobre BigQuery. Asi el gate no
    # depende de credenciales de nube ni del estado de la tabla remota.
    config["datos"]["fuente"] = "csv"

    if not RUTA_MODELO.exists():
        logger.error("No se encontro el modelo en %s. Ejecute primero "
                     "model_training_evaluation.py.", RUTA_MODELO)
        return 1

    artefacto = joblib.load(RUTA_MODELO)
    pipeline = artefacto["pipeline"]

    conjuntos = cargar_conjuntos(config)
    metricas = calcular_metricas(
        pipeline, conjuntos["X_test"], conjuntos["y_test"])

    metricas["pr_auc_minimo"] = float(umbral)
    metricas["aprobado"] = bool(metricas["pr_auc"] >= umbral)

    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    logger.info("Metricas escritas en %s", RUTA_METRICAS)

    for clave, valor in metricas.items():
        logger.info("    %-26s %s", clave, valor)

    if metricas["aprobado"]:
        logger.info("GATE OK: PR-AUC %.4f >= piso %.4f",
                    metricas["pr_auc"], umbral)
        return 0

    logger.error("GATE FALLIDO: PR-AUC %.4f < piso %.4f",
                 metricas["pr_auc"], umbral)
    return 1


if __name__ == "__main__":
    sys.exit(main())
