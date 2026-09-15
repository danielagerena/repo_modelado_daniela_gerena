"""
hueristic_model.py

Modelo heuristico de decision crediticia basado en reglas de negocio
explicitas, interpretables y deterministas.

Las reglas duras prevalecen sobre cualquier prediccion estadistica: cuando
alguna se activa, la solicitud se rechaza sin importar lo que indique el
modelo de aprendizaje automatico. Por eso cada umbral proviene de una fuente
normativa o institucional externa, y no de la separacion que produzca en este
conjunto de datos: un corte ajustado a la muestra no generaliza.

El modelo opera sobre el DataFrame limpio en unidades originales, no sobre la
matriz codificada y escalada que consume el modelo de ML.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

APROBAR = 1
RECHAZAR = 0


@dataclass
class ReglaDura:
    """Condicion que prevalece sobre la prediccion del modelo estadistico.

    Attributes:
        nombre: identificador de la regla, usado en los reportes.
        variable: columna del DataFrame sobre la que opera.
        condicion: funcion que recibe la columna y devuelve una mascara
            booleana con los registros que la regla marca para rechazo.
        descripcion: enunciado de negocio de la regla.
        fuente: norma o referencia institucional que sustenta el umbral.
    """

    nombre: str
    variable: str
    condicion: Callable[[pd.Series], pd.Series]
    descripcion: str
    fuente: str

    def evaluar(self, X: pd.DataFrame) -> pd.Series:
        """Aplica la regla y devuelve que registros marca para rechazo.

        Si la variable no esta presente, la regla no marca a nadie en lugar de
        fallar: permite reutilizar el modelo sobre subconjuntos de columnas.
        """
        if self.variable not in X.columns:
            logger.warning(
                "La regla '%s' se omite: falta la columna %s",
                self.nombre, self.variable,
            )
            return pd.Series(False, index=X.index)

        return self.condicion(X[self.variable]).fillna(False).astype(bool)


def construir_reglas(config: dict[str, Any]) -> list[ReglaDura]:
    """Arma la lista de reglas duras a partir de la configuracion.

    Solo se incluyen las reglas marcadas como activas. Los umbrales se leen
    del config y nunca se escriben en el codigo, de modo que ajustar una
    politica de riesgo no exige modificar el pipeline.

    Args:
        config: configuracion del proyecto.

    Returns:
        Lista de reglas listas para evaluar.
    """
    definiciones = config["reglas_heuristicas"]
    reglas: list[ReglaDura] = []

    if definiciones["mora_vigente"]["activa"]:
        parametros = definiciones["mora_vigente"]
        reglas.append(ReglaDura(
            nombre="mora_vigente",
            variable=parametros["variable"],
            condicion=lambda columna: columna == 1,
            descripcion=parametros["descripcion"],
            fuente=parametros["fuente"],
        ))

    if definiciones["score_alto_riesgo"]["activa"]:
        parametros = definiciones["score_alto_riesgo"]
        categoria = parametros["categoria_rechazo"]
        reglas.append(ReglaDura(
            nombre="score_alto_riesgo",
            variable=parametros["variable"],
            condicion=lambda columna, cat=categoria: columna == cat,
            descripcion=parametros["descripcion"],
            fuente=parametros["fuente"],
        ))

    if definiciones["esfuerzo_pago_excesivo"]["activa"]:
        parametros = definiciones["esfuerzo_pago_excesivo"]
        umbral = parametros["umbral"]
        reglas.append(ReglaDura(
            nombre="esfuerzo_pago_excesivo",
            variable=parametros["variable"],
            condicion=lambda columna, u=umbral: columna > u,
            descripcion=parametros["descripcion"],
            fuente=parametros["fuente"],
        ))

    if definiciones["edad_fuera_de_politica"]["activa"]:
        parametros = definiciones["edad_fuera_de_politica"]
        umbral = parametros["umbral"]
        reglas.append(ReglaDura(
            nombre="edad_fuera_de_politica",
            variable=parametros["variable"],
            condicion=lambda columna, u=umbral: columna > u,
            descripcion=parametros["descripcion"],
            fuente=parametros["fuente"],
        ))

    logger.info("Reglas duras activas: %d", len(reglas))
    return reglas

# ============================================================================
# MODELO HEURISTICO
# ============================================================================
class HeuristicModel(BaseEstimator, ClassifierMixin):
    """Modelo de decision crediticia por reglas duras, compatible con sklearn.

    Las reglas se combinan con logica OR: basta que una se active para
    rechazar la solicitud.

    Los registros que ninguna regla marca quedan indeterminados y se
    derivan al modelo de ml; predict los reporta como aprobados
    unicamente para mantener la interfaz binaria que exige scikit-learn.

    Opera sobre el DataFrame limpio en unidades originales.
    """

    def __init__(self, reglas: list[ReglaDura] | None = None):
        self.reglas = reglas

    def fit(self, X, y=None):
        """No aprende parametros: las reglas provienen de fuentes externas.

        Se conserva por compatibilidad con la interfaz de scikit-learn y para
        registrar las clases observadas.
        """
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.array([RECHAZAR, APROBAR])
        self.reglas_ = self.reglas or []
        return self

    def _matriz_activaciones(self, X: pd.DataFrame) -> pd.DataFrame:
        """Evalua todas las reglas y devuelve una columna booleana por regla."""
        if not self.reglas_:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(
            {regla.nombre: regla.evaluar(X) for regla in self.reglas_},
            index=X.index,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Rechaza cuando alguna regla dura se activa.

        Args:
            X: DataFrame limpio, en unidades originales.

        Returns:
            Arreglo con RECHAZAR (0) donde alguna regla se activo y
            APROBAR (1) en el resto.
        """
        activaciones = self._matriz_activaciones(X)
        if activaciones.empty:
            return np.full(len(X), APROBAR)

        alguna_activa = activaciones.any(axis=1)
        return np.where(alguna_activa, RECHAZAR, APROBAR)

    def explicar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detalla la decision de cada registro y su motivo.

        Es la salida auditable del modelo: permite responder por que se
        rechazo una solicitud concreta, requisito de cualquier politica de
        credito.

        Returns:
            DataFrame con la decision, el estado (RECHAZO o INDETERMINADO),
            los motivos que se activaron y una columna booleana por regla.
        """
        activaciones = self._matriz_activaciones(X)
        alguna_activa = (
            activaciones.any(axis=1)
            if not activaciones.empty
            else pd.Series(False, index=X.index)
        )

        motivos = activaciones.apply(
            lambda fila: ", ".join(fila.index[fila]) or "ninguno", axis=1
        ) if not activaciones.empty else pd.Series("ninguno", index=X.index)

        detalle = pd.DataFrame({
            "decision": np.where(alguna_activa, RECHAZAR, APROBAR),
            "estado": np.where(alguna_activa, "RECHAZO", "INDETERMINADO"),
            "motivos": motivos,
        }, index=X.index)

        return detalle.join(activaciones)

    def cobertura(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reporta cuantos registros marca cada regla, por separado.

        Muestra tambien el solapamiento: la suma de las activaciones
        individuales suele superar el total de rechazos porque un mismo
        registro puede incumplir varias reglas.
        """
        activaciones = self._matriz_activaciones(X)
        total = len(X)

        resumen = pd.DataFrame({
            "registros_marcados": activaciones.sum(),
            "porcentaje": (activaciones.mean() * 100).round(2),
        })
        resumen.loc["TOTAL (union)"] = [
            int(activaciones.any(axis=1).sum()),
            round(activaciones.any(axis=1).mean() * 100, 2),
        ]
        resumen["del_total"] = total
        return resumen

# ============================================================================
# EVALUACION
# ============================================================================
def evaluar_modelo(modelo: HeuristicModel, X: pd.DataFrame,
                   y: pd.Series, conjunto: str = "") -> dict[str, float]:
    """Calcula las metricas del modelo sobre la clase de incumplimiento.

    Las metricas se miden sobre la clase 0 (incumplimiento) porque es el
    evento de interes. Con un desbalance de 95/5, medir la clase mayoritaria
    produce cifras altas sin significado: un modelo que apruebe a todos
    alcanza 95% de exactitud sin detectar un solo incumplimiento.

    Args:
        modelo: modelo heuristico ya ajustado.
        X: DataFrame limpio en unidades originales.
        y: valores reales del objetivo.
        conjunto: etiqueta para el registro de resultados.

    Returns:
        Diccionario con las metricas calculadas.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    predicciones = modelo.predict(X)

    metricas = {
        "exactitud": accuracy_score(y, predicciones),
        "precision_incumplimiento": precision_score(
            y, predicciones, pos_label=RECHAZAR, zero_division=0),
        "recall_incumplimiento": recall_score(
            y, predicciones, pos_label=RECHAZAR, zero_division=0),
        "f1_incumplimiento": f1_score(
            y, predicciones, pos_label=RECHAZAR, zero_division=0),
        "roc_auc": roc_auc_score(y, predicciones),
    }

    matriz = confusion_matrix(y, predicciones, labels=[RECHAZAR, APROBAR])

    logger.info("--- Desempeno del modelo heuristico (%s) ---", conjunto)
    for nombre, valor in metricas.items():
        logger.info("    %-28s %.4f", nombre, valor)
    logger.info("    Matriz de confusion [filas: real 0,1 | columnas: pred 0,1]")
    logger.info("        %s", matriz.tolist())
    logger.info(
        "    Rechazos correctos: %d | Rechazos erroneos: %d | "
        "Incumplimientos no detectados: %d",
        matriz[0, 0], matriz[1, 0], matriz[0, 1],
    )

    return metricas


def evaluar_reglas(modelo: HeuristicModel, X: pd.DataFrame,
                   y: pd.Series) -> pd.DataFrame:
    """Mide el aporte de cada regla por separado.

    Responde la pregunta que decide si una regla merece su lugar: de los
    registros que rechaza, cuantos habrian incumplido de verdad. Una regla
    cuya precision se acerque a la tasa base de incumplimiento no aporta
    informacion: rechaza igual que si se eligiera al azar.

    Args:
        modelo: modelo heuristico ya ajustado.
        X: DataFrame limpio en unidades originales.
        y: valores reales del objetivo.

    Returns:
        DataFrame con una fila por regla, ordenado por precision.
    """
    incumplio = y == RECHAZAR
    tasa_base = incumplio.mean()
    filas = []

    for regla in modelo.reglas_:
        marcados = regla.evaluar(X)
        n_marcados = int(marcados.sum())
        aciertos = int((marcados & incumplio).sum())
        precision = aciertos / n_marcados if n_marcados else 0.0

        filas.append({
            "regla": regla.nombre,
            "marcados": n_marcados,
            "pct_cartera": round(marcados.mean() * 100, 2),
            "incumplieron": aciertos,
            "precision": round(precision, 4),
            "lift": round(precision / tasa_base, 2) if tasa_base else 0.0,
            "cobertura_malos": round(
                aciertos / incumplio.sum(), 4) if incumplio.sum() else 0.0,
        })

    resumen = pd.DataFrame(filas).sort_values("precision", ascending=False)

    logger.info("--- Aporte de cada regla (tasa base de incumplimiento: %.4f) ---",
                tasa_base)
    logger.info("\n%s", resumen.to_string(index=False))
    logger.info(
        "El lift indica cuantas veces mas incumplimiento concentra la regla "
        "frente a la cartera completa. Un lift cercano a 1 equivale a rechazar "
        "al azar."
    )

    return resumen


def construir_modelo(config: dict[str, Any]) -> HeuristicModel:
    """Instancia el modelo heuristico con las reglas activas del config."""
    return HeuristicModel(reglas=construir_reglas(config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from ft_engineering import cargar_config, generar_datasets

    config = cargar_config()
    datos = generar_datasets(config)

    objetivo = config["datos"]["columna_objetivo"]
    limpios = datos["datos_limpios"]

    # El heuristico se evalua sobre los mismos registros que el modelo de ML,
    # pero en unidades originales y sin balancear.
    X_train_crudo = limpios.loc[datos["X_train"].index].drop(columns=[objetivo])
    X_test_crudo = limpios.loc[datos["X_test"].index].drop(columns=[objetivo])

    modelo = construir_modelo(config)
    modelo.fit(X_train_crudo, datos["y_train"])

    logger.info("\n%s", modelo.cobertura(X_train_crudo).to_string())

    evaluar_reglas(modelo, X_train_crudo, datos["y_train"])
    evaluar_modelo(modelo, X_train_crudo, datos["y_train"], conjunto="entrenamiento")
    evaluar_modelo(modelo, X_test_crudo, datos["y_test"], conjunto="evaluacion")

    ejemplos = modelo.explicar(X_test_crudo)
    rechazados = ejemplos[ejemplos["estado"] == "RECHAZO"]
    logger.info("Rechazos en evaluacion: %d de %d", len(rechazados), len(ejemplos))
    if not rechazados.empty:
        logger.info("Ejemplo de decisiones auditables:\n%s",
                    rechazados[["estado", "motivos"]].head(5).to_string())