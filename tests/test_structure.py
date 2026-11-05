"""test_structure.py

Prueba de estructura para el pipeline de CI (Jenkins).

Valida que el repositorio tenga las carpetas y archivos clave antes de
cualquier ejecucion de modelo. Es una red de seguridad barata: si una
refactorizacion mueve o borra un archivo esencial, el build falla aqui con un
mensaje claro, en lugar de reventar mas adelante con un error oscuro.

Termina con codigo de salida 0 si todo esta en su lugar, y 1 si falta algo.
Parte de los archivos se leen del propio `config.json` (ruta de datos), para
que la prueba siga la filosofia config-driven del proyecto.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

# Carpetas y archivos que deben existir siempre.
RUTAS_REQUERIDAS = [
    "mlops_pipeline",
    "mlops_pipeline/config.json",
    "mlops_pipeline/modelo_final.joblib",
    "mlops_pipeline/src",
    "mlops_pipeline/src/ft_engineering.py",
    "mlops_pipeline/src/model_training_evaluation.py",
    "mlops_pipeline/src/model_deploy.py",
    "mlops_pipeline/src/evaluate.py",
    "requirements.txt",
]


def _ruta_datos_desde_config() -> str | None:
    """Lee la ruta del archivo de datos declarada en config.json, si aplica."""
    import json

    config_path = RAIZ_PROYECTO / "mlops_pipeline" / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    # Solo se exige el CSV cuando la fuente declarada es un archivo local.
    datos = config.get("datos", {})
    if datos.get("fuente") == "bigquery" and not (RAIZ_PROYECTO / datos.get("ruta", "")).exists():
        # La fuente es BigQuery; el CSV local es opcional.
        return None
    return datos.get("ruta")


def validar() -> list[str]:
    """Devuelve la lista de rutas faltantes (vacia si todo esta correcto)."""
    faltantes = []
    for relativa in RUTAS_REQUERIDAS:
        if not (RAIZ_PROYECTO / relativa).exists():
            faltantes.append(relativa)

    ruta_datos = _ruta_datos_desde_config()
    if ruta_datos and not (RAIZ_PROYECTO / ruta_datos).exists():
        faltantes.append(ruta_datos)

    return faltantes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    faltantes = validar()
    if faltantes:
        for ruta in faltantes:
            logger.error("FALTA: %s", ruta)
        logger.error("Estructura invalida: %d ruta(s) faltante(s).", len(faltantes))
        return 1

    logger.info("Estructura OK: %d rutas verificadas.", len(RUTAS_REQUERIDAS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
