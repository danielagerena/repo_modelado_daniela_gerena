"""
model_training_evaluation.py

Entrenamiento, comparacion y evaluacion de modelos de clasificacion para la
prediccion de incumplimiento en cartera de consumo.

Unifica las etapas de entrenamiento y evaluacion en un solo componente, por
solicitud del docente. El proceso es:

    1. Validacion cruzada repetida de siete modelos sobre el 70% de los datos
    2. Comparacion en desempeño, consistencia y escalabilidad
    3. Seleccion de tres finalistas mediante ANOVA y Tukey HSD
    4. Optimizacion de hiperparametros
    5. Evaluacion del modelo ganador sobre el 30% reservado
    6. Persistencia del pipeline completo para despliegue

El preprocesamiento y el balanceo viven DENTRO del pipeline de cada modelo,
no se aplican antes. De lo contrario, los estadisticos de imputacion y los
registros sinteticos de SMOTE se calcularian con informacion de los pliegues
de validacion, inflando las metricas.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (  # noqa: E402
    cargar_config,
    construir_preprocesador,
    generar_datasets,
)

logger = logging.getLogger(__name__)

INCUMPLE = 0
CUMPLE = 1


def cargar_conjuntos(config: dict[str, Any]) -> dict[str, Any]:
    """Recupera los conjuntos de entrenamiento y evaluacion sin codificar.

    La particion 70/30 se define en ft_engineering; aqui solo se recuperan
    los mismos registros en unidades originales, porque el preprocesamiento
    se aplicara dentro del pipeline de cada modelo.

    Returns:
        Diccionario con X_train, X_test, y_train, y_test sin transformar.
    """
    datos = generar_datasets(config)
    objetivo = config["datos"]["columna_objetivo"]
    limpios = datos["datos_limpios"]

    predictoras = limpios.drop(columns=[objetivo])

    conjuntos = {
        "X_train": predictoras.loc[datos["X_train"].index],
        "X_test": predictoras.loc[datos["X_test"].index],
        "y_train": datos["y_train"],
        "y_test": datos["y_test"],
    }

    logger.info(
        "Entrenamiento: %d registros | Evaluacion: %d registros",
        len(conjuntos["X_train"]), len(conjuntos["X_test"]),
    )
    logger.info(
        "Tasa de incumplimiento — entrenamiento: %.4f | evaluacion: %.4f",
        (conjuntos["y_train"] == INCUMPLE).mean(),
        (conjuntos["y_test"] == INCUMPLE).mean(),
    )
    return conjuntos


def build_model(clasificador, config: dict[str, Any],
                balancear: bool = True) -> PipelineImb:
    """Arma el pipeline completo de un modelo.

    El pipeline encadena preprocesamiento, balanceo y clasificador. Al pasarlo
    a la validacion cruzada, cada pliegue ajusta el imputador, el escalador y
    los codificadores con sus propios datos de entrenamiento, y genera sus
    propios registros sinteticos. Asi ningun pliegue de validacion contribuye
    a los parametros con los que se le evalua.

    SMOTE se omite en prediccion de forma automatica: los pasos de remuestreo
    de imbalanced-learn solo actuan durante el ajuste.

    Args:
        clasificador: estimador de scikit-learn ya instanciado.
        config: configuracion del proyecto.
        balancear: si se incluye el paso de SMOTE.

    Returns:
        Pipeline listo para ajustar o para validacion cruzada.
    """
    pasos = [("preprocesador", construir_preprocesador(config))]

    if balancear:
        pasos.append((
            "balanceo",
            SMOTE(
                sampling_strategy=config["balanceo"].get("sampling_strategy", 1.0),
                random_state=config["particion"]["random_state"],
            ),
        ))

    pasos.append(("modelo", clasificador))
    return PipelineImb(pasos)


def summarize_classification(y_real, y_pred, y_proba=None,
                             nombre: str = "") -> dict[str, float]:
    """Resume el desempeno de un modelo sobre la clase de incumplimiento.

    Las metricas se calculan sobre la clase 0 porque es el evento de interes.
    Con un desbalance de 95/5, medir la clase mayoritaria produce cifras altas
    sin significado: aprobar a todos alcanza 95% de exactitud sin detectar un
    solo incumplimiento.

    Args:
        y_real: valores observados.
        y_pred: predicciones binarias.
        y_proba: probabilidad de incumplimiento. Necesaria para PR-AUC y
            ROC-AUC, que evaluan la capacidad de ordenar clientes por riesgo
            con independencia del umbral de corte.
        nombre: etiqueta del modelo para el registro.

    Returns:
        Diccionario con las metricas calculadas.
    """
    resumen = {
        "precision_incumplimiento": precision_score(
            y_real, y_pred, pos_label=INCUMPLE, zero_division=0),
        "recall_incumplimiento": recall_score(
            y_real, y_pred, pos_label=INCUMPLE, zero_division=0),
        "f1_incumplimiento": f1_score(
            y_real, y_pred, pos_label=INCUMPLE, zero_division=0),
        "exactitud": (y_real == y_pred).mean(),
    }

    if y_proba is not None:
        incumplio = (y_real == INCUMPLE).astype(int)
        resumen["pr_auc"] = average_precision_score(incumplio, y_proba)
        resumen["roc_auc"] = roc_auc_score(incumplio, y_proba)

    if nombre:
        logger.info("--- %s ---", nombre)
        for metrica, valor in resumen.items():
            logger.info("    %-28s %.4f", metrica, valor)
        logger.info("    Matriz de confusion [real 0,1 x pred 0,1]: %s",
                    confusion_matrix(y_real, y_pred,
                                     labels=[INCUMPLE, CUMPLE]).tolist())

    return resumen
# ============================================================================
# MODELOS CANDIDATOS
# ============================================================================
def definir_modelos(config: dict[str, Any]) -> dict[str, Any]:
    """Instancia los siete modelos candidatos.

    Cuatro metodos supervisados de familias distintas y tres ensambles con
    mecanismos distintos. La diversidad es deliberada: comparar tres variantes
    del mismo algoritmo no aporta informacion sobre que enfoque conviene.

        Regresion logistica  lineal, estandar del scoring crediticio
        Arbol de decision    basado en reglas, comparable con el heuristico
        KNN                  basado en distancia
        Red neuronal         aproximador no lineal
        Random Forest        ensamble por bagging
        HistGradientBoosting ensamble por boosting con histogramas
        AdaBoost             ensamble por boosting adaptativo

    La profundidad del arbol y el numero de vecinos se dejan en valores
    conservadores: la optimizacion de hiperparametros ocurre despues, solo
    sobre los finalistas.
    """
    semilla = config["particion"]["random_state"]

    return {
        "regresion_logistica": LogisticRegression(
            max_iter=1000, random_state=semilla),
        "arbol_decision": DecisionTreeClassifier(
            max_depth=5, random_state=semilla),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "red_neuronal": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000,
            early_stopping=True, random_state=semilla),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=semilla),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=semilla),
        "adaboost": AdaBoostClassifier(
            n_estimators=100, random_state=semilla),
    }


# ============================================================================
# VALIDACION CRUZADA
# ============================================================================
def _pr_auc_incumplimiento(estimador, X, y) -> float:
    """Calcula el PR-AUC tomando el incumplimiento como evento positivo.

    predict_proba devuelve las probabilidades en el orden de classes_, que
    aqui es [0, 1]. La probabilidad de incumplir es la primera columna; usar
    la segunda por costumbre invertiria la metrica.
    """
    probabilidades = estimador.predict_proba(X)
    indice = list(estimador.classes_).index(INCUMPLE)
    return average_precision_score(
        (y == INCUMPLE).astype(int), probabilidades[:, indice]
    )


def validar_modelos(modelos: dict[str, Any], X_train: pd.DataFrame,
                    y_train: pd.Series, config: dict[str, Any],
                    repeticiones: int = 3, pliegues: int = 5) -> pd.DataFrame:
    """Evalua cada modelo con validacion cruzada estratificada repetida.

    Se usan repeticiones porque cinco mediciones por modelo no bastan para
    contrastar diferencias con ANOVA. Con cinco pliegues y tres repeticiones
    se obtienen quince mediciones por modelo.

    La estratificacion es indispensable: con 4,7% de incumplimiento, una
    particion aleatoria podria dejar pliegues casi sin casos positivos.

    Args:
        modelos: diccionario nombre -> estimador.
        X_train: predictoras sin codificar.
        y_train: objetivo.
        config: configuracion del proyecto.
        repeticiones: veces que se repite el esquema completo.
        pliegues: particiones por repeticion.

    Returns:
        DataFrame con una fila por modelo y medicion, en formato largo.
    """
    particion = RepeatedStratifiedKFold(
        n_splits=pliegues,
        n_repeats=repeticiones,
        random_state=config["particion"]["random_state"],
    )

    metricas = {
        "pr_auc": "average_precision",
        "recall_incumplimiento": "recall",
        "precision_incumplimiento": "precision",
        "f1_incumplimiento": "f1",
        "roc_auc": "roc_auc",
    }
    # Las metricas de sklearn miden la clase 1 por defecto. Como el objetivo
    # ya esta codificado con 0 = incumplimiento, se invierte el objetivo para
    # que las cadenas estandar apunten al evento de interes.
    y_invertido = (y_train == INCUMPLE).astype(int)

    resultados = []

    for nombre, clasificador in modelos.items():
        logger.info("Validando %s ...", nombre)
        inicio = time.perf_counter()

        pipeline = build_model(clasificador, config, balancear=True)
        puntajes = cross_validate(
            pipeline, X_train, y_invertido,
            cv=particion, scoring=metricas,
            return_train_score=False, n_jobs=1,
        )

        duracion = time.perf_counter() - inicio
        logger.info("    %s completado en %.1f s", nombre, duracion)

        for indice in range(len(puntajes["fit_time"])):
            fila = {"modelo": nombre, "medicion": indice}
            for metrica in metricas:
                fila[metrica] = puntajes[f"test_{metrica}"][indice]
            fila["tiempo_ajuste"] = puntajes["fit_time"][indice]
            fila["tiempo_prediccion"] = puntajes["score_time"][indice]
            resultados.append(fila)

    return pd.DataFrame(resultados)


def resumir_comparacion(mediciones: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla resumen en desempeno, consistencia y escalabilidad.

    Las tres dimensiones son las que exige el enunciado:

        Performance   media del PR-AUC entre las mediciones
        Consistency   desviacion estandar entre mediciones. Un modelo con
                      media alta y varianza alta no es confiable en produccion
        Scalability   tiempo de ajuste y de prediccion

    Returns:
        DataFrame ordenado por PR-AUC medio, de mayor a menor.
    """
    resumen = mediciones.groupby("modelo").agg(
        pr_auc_medio=("pr_auc", "mean"),
        pr_auc_desv=("pr_auc", "std"),
        recall_medio=("recall_incumplimiento", "mean"),
        precision_media=("precision_incumplimiento", "mean"),
        f1_medio=("f1_incumplimiento", "mean"),
        roc_auc_medio=("roc_auc", "mean"),
        ajuste_seg=("tiempo_ajuste", "mean"),
        prediccion_seg=("tiempo_prediccion", "mean"),
    ).round(4)

    resumen["coef_variacion"] = (
        resumen["pr_auc_desv"] / resumen["pr_auc_medio"]
    ).round(4)

    resumen = resumen.sort_values("pr_auc_medio", ascending=False)

    logger.info("--- Comparacion de modelos ---")
    logger.info("\n%s", resumen.to_string())
    return resumen

# ============================================================================
# SELECCION ESTADISTICA DE FINALISTAS
# ============================================================================
def comparar_estadisticamente(mediciones: pd.DataFrame,
                              metrica: str = "pr_auc",
                              alfa: float = 0.05) -> dict[str, Any]:
    """Contrasta si las diferencias entre modelos son significativas.

    Primero un ANOVA de un factor sobre las mediciones de la validacion
    cruzada. Si detecta diferencias, un Tukey HSD identifica que pares de
    modelos difieren, corrigiendo por las comparaciones multiples: contrastar
    veintiun pares al 5% produciria falsos positivos por puro azar.

    Limitacion conocida: el ANOVA supone observaciones independientes, y los
    pliegues de una validacion cruzada comparten datos entre si. El p-valor
    resultante es optimista. El contraste se aplica porque el enunciado lo
    exige y porque sigue siendo util como criterio de ordenamiento, pero no
    debe leerse como una prueba formal de superioridad.

    Args:
        mediciones: resultados en formato largo de validar_modelos.
        metrica: columna sobre la que se contrasta.
        alfa: nivel de significancia.

    Returns:
        Diccionario con el estadistico F, el p-valor, la tabla de Tukey y si
        se detectaron diferencias.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    grupos = [
        grupo[metrica].values
        for _, grupo in mediciones.groupby("modelo")
    ]
    estadistico_f, p_valor = f_oneway(*grupos)

    logger.info("--- ANOVA sobre %s ---", metrica)
    logger.info("    F = %.4f | p = %.6f", estadistico_f, p_valor)

    hay_diferencias = p_valor < alfa
    if hay_diferencias:
        logger.info("    Se detectan diferencias significativas entre modelos.")
    else:
        logger.info(
            "    No se detectan diferencias significativas. La seleccion "
            "debera apoyarse en criterios secundarios."
        )

    tukey = pairwise_tukeyhsd(
        endog=mediciones[metrica],
        groups=mediciones["modelo"],
        alpha=alfa,
    )
    logger.info("--- Tukey HSD ---")
    logger.info("\n%s", tukey)

    return {
        "estadistico_f": estadistico_f,
        "p_valor": p_valor,
        "hay_diferencias": hay_diferencias,
        "tukey": tukey,
        "tukey_df": pd.DataFrame(
            tukey.summary().data[1:], columns=tukey.summary().data[0]
        ),
    }


def seleccionar_finalistas(resumen: pd.DataFrame, contraste: dict[str, Any],
                           n_finalistas: int = 3) -> list[str]:
    """Elige los modelos que pasan a la optimizacion de hiperparametros.

    El criterio principal es el PR-AUC medio. Cuando el ANOVA no detecta
    diferencias significativas, o cuando dos modelos resultan
    estadisticamente indistinguibles segun Tukey, se recurre a criterios
    secundarios en este orden:

        1. Menor coeficiente de variacion (mayor consistencia)
        2. Menor tiempo de ajuste (mayor escalabilidad)

    Preferir el modelo mas estable y mas rapido ante un empate estadistico
    evita elegir por diferencias que son ruido de muestreo.

    Args:
        resumen: tabla comparativa ordenada por PR-AUC.
        contraste: salida de comparar_estadisticamente.
        n_finalistas: cuantos modelos avanzan.

    Returns:
        Lista con los nombres de los modelos seleccionados.
    """
    if not contraste["hay_diferencias"]:
        logger.warning(
            "Sin diferencias significativas: se seleccionan los %d modelos "
            "mas consistentes entre los de mejor desempeno.", n_finalistas,
        )
        candidatos = resumen.head(max(n_finalistas + 2, len(resumen)))
        seleccionados = (
            candidatos.sort_values(["coef_variacion", "ajuste_seg"])
            .head(n_finalistas).index.tolist()
        )
    else:
        seleccionados = resumen.head(n_finalistas).index.tolist()

    logger.info("Finalistas seleccionados: %s", seleccionados)
    for nombre in seleccionados:
        fila = resumen.loc[nombre]
        logger.info(
            "    %-24s PR-AUC %.4f (desv %.4f, CV %.4f) | ajuste %.2f s",
            nombre, fila["pr_auc_medio"], fila["pr_auc_desv"],
            fila["coef_variacion"], fila["ajuste_seg"],
        )

    return seleccionados

# ============================================================================
# OPTIMIZACION DE HIPERPARAMETROS
# ============================================================================
def definir_grids() -> dict[str, dict[str, list]]:
    """Rejillas de busqueda para GridSearch, por modelo.

    Los rangos son deliberadamente acotados. Cada combinacion se evalua con
    validacion cruzada completa, asi que una rejilla de cien combinaciones
    supone cientos de ajustes; ampliarla sin criterio multiplica el tiempo sin
    mejorar el resultado.

    Los prefijos 'modelo__' indican que el parametro pertenece al paso
    'modelo' del pipeline, no al preprocesador ni al balanceo.
    """
    return {
        "regresion_logistica": {
            "modelo__C": [0.01, 0.1, 1.0, 10.0],
            "modelo__solver": ["lbfgs", "liblinear"],
        },
        "arbol_decision": {
            "modelo__max_depth": [3, 5, 8, 12],
            "modelo__min_samples_leaf": [10, 30, 50],
            "modelo__criterion": ["gini", "entropy"],
        },
        "knn": {
            "modelo__n_neighbors": [5, 15, 25, 45],
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        },
        "red_neuronal": {
            "modelo__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
            "modelo__alpha": [0.0001, 0.01, 0.1],
        },
        "random_forest": {
            "modelo__n_estimators": [100, 300],
            "modelo__max_depth": [5, 10, 20],
            "modelo__min_samples_leaf": [1, 10, 30],
        },
        "hist_gradient_boosting": {
            "modelo__max_iter": [100, 300],
            "modelo__learning_rate": [0.01, 0.1],
            "modelo__max_leaf_nodes": [15, 31],
        },
        "adaboost": {
            "modelo__n_estimators": [50, 100, 200],
            "modelo__learning_rate": [0.1, 0.5, 1.0],
        },
    }


def optimizar_con_grid(nombre: str, clasificador, X_train, y_train,
                       config: dict[str, Any], pliegues: int = 5):
    """Busca los mejores hiperparametros probando todas las combinaciones.

    La busqueda optimiza PR-AUC y usa validacion cruzada interna sobre el
    conjunto de entrenamiento. El conjunto de evaluacion no participa: si se
    usara para elegir hiperparametros, dejaria de ser una estimacion honesta
    del desempeno.

    Returns:
        Tupla con el mejor pipeline, sus parametros y su puntaje.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    rejilla = definir_grids()[nombre]
    pipeline = build_model(clasificador, config, balancear=True)

    busqueda = GridSearchCV(
        pipeline, rejilla,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=pliegues, shuffle=True,
                           random_state=config["particion"]["random_state"]),
        n_jobs=-1, refit=True,
    )

    inicio = time.perf_counter()
    busqueda.fit(X_train, (y_train == INCUMPLE).astype(int))
    duracion = time.perf_counter() - inicio

    logger.info("GridSearch en %s (%.1f s)", nombre, duracion)
    logger.info("    Mejor PR-AUC: %.4f", busqueda.best_score_)
    logger.info("    Parametros:   %s", busqueda.best_params_)

    return busqueda.best_estimator_, busqueda.best_params_, busqueda.best_score_


def optimizar_con_bayesiana(nombre: str, clasificador, X_train, y_train,
                            config: dict[str, Any], n_intentos: int = 40,
                            pliegues: int = 5):
    """Busca hiperparametros con optimizacion bayesiana.

    A diferencia de GridSearch, que prueba combinaciones fijas sin aprender de
    los resultados, la busqueda bayesiana construye un modelo de que zonas del
    espacio son prometedoras y concentra los intentos alli. Encuentra buenas
    configuraciones en menos evaluaciones y explora valores continuos que una
    rejilla discreta no contiene.

    Returns:
        Tupla con el pipeline ajustado, los parametros y el puntaje.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    semilla = config["particion"]["random_state"]
    y_binario = (y_train == INCUMPLE).astype(int)
    particion = StratifiedKFold(n_splits=pliegues, shuffle=True,
                                random_state=semilla)

    def espacio_busqueda(intento) -> dict[str, Any]:
        """Define el rango de cada hiperparametro segun el modelo."""
        if nombre == "regresion_logistica":
            return {"modelo__C": intento.suggest_float("C", 1e-3, 1e2, log=True)}
        if nombre == "arbol_decision":
            return {
                "modelo__max_depth": intento.suggest_int("max_depth", 2, 20),
                "modelo__min_samples_leaf": intento.suggest_int(
                    "min_samples_leaf", 5, 100),
            }
        if nombre == "knn":
            return {
                "modelo__n_neighbors": intento.suggest_int("n_neighbors", 3, 60),
                "modelo__weights": intento.suggest_categorical(
                    "weights", ["uniform", "distance"]),
            }
        if nombre == "red_neuronal":
            return {
                "modelo__alpha": intento.suggest_float("alpha", 1e-5, 1e0, log=True),
                "modelo__learning_rate_init": intento.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True),
            }
        if nombre == "random_forest":
            return {
                "modelo__n_estimators": intento.suggest_int("n_estimators", 50, 400),
                "modelo__max_depth": intento.suggest_int("max_depth", 3, 25),
                "modelo__min_samples_leaf": intento.suggest_int(
                    "min_samples_leaf", 1, 50),
            }
        if nombre == "hist_gradient_boosting":
            return {
                "modelo__learning_rate": intento.suggest_float(
                    "learning_rate", 1e-3, 3e-1, log=True),
                "modelo__max_iter": intento.suggest_int("max_iter", 50, 400),
                "modelo__max_leaf_nodes": intento.suggest_int(
                    "max_leaf_nodes", 7, 63),
            }
        if nombre == "adaboost":
            return {
                "modelo__n_estimators": intento.suggest_int("n_estimators", 30, 300),
                "modelo__learning_rate": intento.suggest_float(
                    "learning_rate", 1e-2, 2.0, log=True),
            }
        return {}

    def objetivo(intento) -> float:
        pipeline = build_model(clasificador, config, balancear=True)
        pipeline.set_params(**espacio_busqueda(intento))
        puntajes = cross_val_score(
            pipeline, X_train, y_binario,
            scoring="average_precision", cv=particion, n_jobs=-1,
        )
        return float(np.mean(puntajes))

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
    )

    inicio = time.perf_counter()
    estudio.optimize(objetivo, n_trials=n_intentos, show_progress_bar=False)
    duracion = time.perf_counter() - inicio

    logger.info("Optimizacion bayesiana en %s (%.1f s, %d intentos)",
                nombre, duracion, n_intentos)
    logger.info("    Mejor PR-AUC: %.4f", estudio.best_value)
    logger.info("    Parametros:   %s", estudio.best_params)

    mejor_pipeline = build_model(clasificador, config, balancear=True)
    mejor_pipeline.set_params(**espacio_busqueda(estudio.best_trial))
    mejor_pipeline.fit(X_train, y_binario)

    return mejor_pipeline, estudio.best_params, estudio.best_value


def optimizar_finalistas(finalistas: list[str], modelos: dict[str, Any],
                         X_train, y_train, config: dict[str, Any]) -> pd.DataFrame:
    """Aplica ambos metodos de busqueda a cada finalista y los compara.

    Ejecutar los dos permite verificar si la busqueda bayesiana encuentra
    configuraciones que la rejilla discreta no contenia, que es su ventaja
    esperada sobre parametros continuos.

    Returns:
        DataFrame con el resultado de cada combinacion modelo-metodo,
        ordenado por puntaje.
    """
    resultados = []

    for nombre in finalistas:
        clasificador = modelos[nombre]

        pipeline_grid, params_grid, score_grid = optimizar_con_grid(
            nombre, clasificador, X_train, y_train, config)
        resultados.append({
            "modelo": nombre, "metodo": "grid_search",
            "pr_auc": score_grid, "parametros": params_grid,
            "pipeline": pipeline_grid,
        })

        pipeline_bayes, params_bayes, score_bayes = optimizar_con_bayesiana(
            nombre, clasificador, X_train, y_train, config)
        resultados.append({
            "modelo": nombre, "metodo": "bayesiana",
            "pr_auc": score_bayes, "parametros": params_bayes,
            "pipeline": pipeline_bayes,
        })

    tabla = pd.DataFrame(resultados).sort_values("pr_auc", ascending=False)

    logger.info("--- Resultado de la optimizacion ---")
    logger.info("\n%s", tabla[["modelo", "metodo", "pr_auc"]].to_string(index=False))

    return tabla

# ============================================================================
# EVALUACION FINAL
# ============================================================================
def evaluar_en_test(pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                    nombre: str) -> dict[str, float]:
    """Evalua el modelo ganador sobre el conjunto reservado.

    Este conjunto se usa una sola vez, al final. Si participara en la
    seleccion de modelos o de hiperparametros, dejaria de ser una estimacion
    honesta del desempeno esperado en produccion.

    Returns:
        Diccionario con las metricas del modelo sobre datos no vistos.
    """
    y_binario = (y_test == INCUMPLE).astype(int)
    probabilidades = pipeline.predict_proba(X_test)[:, 1]
    predicciones = pipeline.predict(X_test)

    metricas = {
        "pr_auc": average_precision_score(y_binario, probabilidades),
        "roc_auc": roc_auc_score(y_binario, probabilidades),
        "precision_incumplimiento": precision_score(
            y_binario, predicciones, zero_division=0),
        "recall_incumplimiento": recall_score(
            y_binario, predicciones, zero_division=0),
        "f1_incumplimiento": f1_score(y_binario, predicciones, zero_division=0),
        "exactitud": (y_binario == predicciones).mean(),
    }

    logger.info("--- Evaluacion final: %s ---", nombre)
    for metrica, valor in metricas.items():
        logger.info("    %-28s %.4f", metrica, valor)
    logger.info("    Matriz de confusion [real 0,1 x pred 0,1]:\n%s",
                confusion_matrix(y_binario, predicciones))
    logger.info("\n%s", classification_report(
        y_binario, predicciones,
        target_names=["cumple", "incumple"], zero_division=0))

    return metricas


def evaluar_heuristico_como_linea_base(X_test: pd.DataFrame, y_test: pd.Series,
                                       config: dict[str, Any]) -> dict[str, float]:
    """Mide el modelo de reglas sobre el mismo conjunto de evaluacion.

    No compite con los modelos estadisticos: es una capa de politica que
    prevalece sobre ellos. Se incluye en la tabla como referencia para
    responder que aporta el aprendizaje automatico por encima de un sistema
    de reglas de negocio.
    """
    from hueristic_model import construir_modelo

    modelo = construir_modelo(config)
    modelo.fit(X_test, y_test)
    predicciones = modelo.predict(X_test)

    y_binario = (y_test == INCUMPLE).astype(int)
    pred_binaria = (predicciones == INCUMPLE).astype(int)

    metricas = {
        "pr_auc": np.nan,  # el heuristico no produce probabilidades
        "roc_auc": roc_auc_score(y_binario, pred_binaria),
        "precision_incumplimiento": precision_score(
            y_binario, pred_binaria, zero_division=0),
        "recall_incumplimiento": recall_score(
            y_binario, pred_binaria, zero_division=0),
        "f1_incumplimiento": f1_score(y_binario, pred_binaria, zero_division=0),
        "exactitud": (y_binario == pred_binaria).mean(),
    }

    logger.info("--- Linea base: modelo heuristico ---")
    for metrica, valor in metricas.items():
        logger.info("    %-28s %.4f", metrica, valor)

    return metricas


# ============================================================================
# GRAFICOS COMPARATIVOS
# ============================================================================
def generar_graficos(mediciones: pd.DataFrame, resumen: pd.DataFrame,
                     pipeline_final, X_test, y_test,
                     ruta_salida: Path) -> None:
    """Produce los graficos comparativos y de diagnostico.

    Se incluyen curvas de precision-recall y de calibracion en lugar de los
    diagnosticos de residuales propios de la regresion: en clasificacion
    binaria el residuo solo toma dos valores y su grafico no informa nada.
    La calibracion responde la pregunta equivalente: cuando el modelo estima
    un riesgo del veinte por ciento, incumple efectivamente ese veinte por
    ciento del grupo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    ruta_salida.mkdir(parents=True, exist_ok=True)
    y_binario = (y_test == INCUMPLE).astype(int)
    probabilidades = pipeline_final.predict_proba(X_test)[:, 1]

    # 1. Dispersion del PR-AUC entre mediciones (desempeno y consistencia)
    figura, eje = plt.subplots(figsize=(10, 5))
    orden = resumen.index.tolist()
    datos = [mediciones.loc[mediciones["modelo"] == m, "pr_auc"] for m in orden]
    eje.boxplot(datos, tick_labels=orden)
    eje.set_ylabel("PR-AUC")
    eje.set_title("Desempeno y consistencia por modelo (validacion cruzada)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    figura.savefig(ruta_salida / "comparacion_modelos.png", dpi=120)
    plt.close(figura)

    # 2. Desempeno frente a costo computacional (escalabilidad)
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.scatter(resumen["ajuste_seg"], resumen["pr_auc_medio"])
    for nombre in resumen.index:
        eje.annotate(nombre,
                     (resumen.loc[nombre, "ajuste_seg"],
                      resumen.loc[nombre, "pr_auc_medio"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    eje.set_xlabel("Tiempo de ajuste (s)")
    eje.set_ylabel("PR-AUC medio")
    eje.set_title("Desempeno frente a escalabilidad")
    plt.tight_layout()
    figura.savefig(ruta_salida / "desempeno_vs_costo.png", dpi=120)
    plt.close(figura)

    # 3. Curva precision-recall del modelo ganador
    precision, recall, _ = precision_recall_curve(y_binario, probabilidades)
    tasa_base = y_binario.mean()
    figura, eje = plt.subplots(figsize=(7, 5))
    eje.plot(recall, precision, label="Modelo ganador")
    eje.axhline(tasa_base, linestyle="--",
                label=f"Clasificador aleatorio ({tasa_base:.3f})")
    eje.set_xlabel("Recall (incumplimientos detectados)")
    eje.set_ylabel("Precision (rechazos acertados)")
    eje.set_title("Curva precision-recall")
    eje.legend()
    plt.tight_layout()
    figura.savefig(ruta_salida / "curva_precision_recall.png", dpi=120)
    plt.close(figura)

    # 4. Calibracion de las probabilidades
    fraccion_real, prob_predicha = calibration_curve(
        y_binario, probabilidades, n_bins=10, strategy="quantile")
    figura, eje = plt.subplots(figsize=(7, 5))
    eje.plot(prob_predicha, fraccion_real, marker="o", label="Modelo")
    eje.plot([0, 1], [0, 1], linestyle="--", label="Calibracion perfecta")
    eje.set_xlabel("Probabilidad estimada de incumplimiento")
    eje.set_ylabel("Frecuencia observada")
    eje.set_title("Curva de calibracion")
    eje.legend()
    plt.tight_layout()
    figura.savefig(ruta_salida / "curva_calibracion.png", dpi=120)
    plt.close(figura)

    logger.info("Graficos guardados en %s", ruta_salida)


# ============================================================================
# PERSISTENCIA
# ============================================================================
def guardar_modelo(pipeline, metadatos: dict[str, Any], ruta: Path) -> None:
    """Persiste el pipeline completo junto con su trazabilidad.

    Se guarda el pipeline entero, no solo el clasificador: incluye el
    preprocesamiento, de modo que el endpoint recibe registros limpios en
    unidades originales y no necesita replicar la imputacion, el escalado ni
    la codificacion. Replicar esas operaciones a mano es una fuente habitual
    de predicciones silenciosamente erroneas en produccion.

    El paso de balanceo no interviene en prediccion: los remuestreadores de
    imbalanced-learn solo actuan durante el ajuste.
    """
    import joblib

    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadatos": metadatos}, ruta)
    logger.info("Modelo guardado en %s", ruta)

def seleccionar_ganador(optimizados: pd.DataFrame, resumen: pd.DataFrame,
                        tolerancia: float = 0.01):
    """Elige el modelo final aplicando una regla de desempate explicita.

    Tomar el maximo puntaje sin mas equivale a elegir por diferencias que
    pueden estar dentro del ruido de la validacion cruzada. Cuando varios
    modelos quedan dentro de la tolerancia del mejor, se prefiere el que la
    validacion cruzada mostro mas consistente y mas rapido de ajustar, que
    son las dimensiones de consistencia y escalabilidad del enunciado.

    Args:
        optimizados: resultados de la optimizacion, ordenados por puntaje.
        resumen: tabla comparativa de la validacion cruzada.
        tolerancia: diferencia de PR-AUC por debajo de la cual dos modelos se
            consideran empatados.
    """
    mejor_puntaje = optimizados["pr_auc"].max()
    empatados = optimizados[
        optimizados["pr_auc"] >= mejor_puntaje - tolerancia
    ].copy()

    if len(empatados) == 1:
        return empatados.iloc[0]

    empatados["cv_pr_auc"] = empatados["modelo"].map(resumen["pr_auc_medio"])
    empatados["cv_tiempo"] = empatados["modelo"].map(resumen["ajuste_seg"])

    logger.info(
        "Empate tecnico dentro de %.3f de PR-AUC entre %d configuraciones. "
        "La optimizacion evalua cada modelo muchas veces sobre la misma "
        "particion, asi que su puntaje esta sesgado al alza para los modelos "
        "con mas hiperparametros. Se desempata por el desempeno en validacion "
        "cruzada, que no participo de esa busqueda, y por tiempo de ajuste.",
        tolerancia, len(empatados),
    )
    logger.info("\n%s", empatados[
        ["modelo", "metodo", "pr_auc", "cv_pr_auc", "cv_tiempo"]
    ].to_string(index=False))

    return empatados.sort_values(
        ["cv_pr_auc", "cv_tiempo"], ascending=[False, True]
    ).iloc[0]

# ============================================================================
# UMBRAL DE CORTE
# ============================================================================
def analizar_umbrales(pipeline, X: pd.DataFrame, y: pd.Series,
                      config: dict[str, Any],
                      pasos: int = 19) -> pd.DataFrame:
    """Evalua el efecto de cada umbral de corte sobre el negocio.

    El umbral por defecto de 0,5 supone que ambos errores cuestan lo mismo y
    que las probabilidades estan calibradas. Ninguna de las dos cosas se
    cumple aqui: rechazar a un cliente viable cuesta el margen del credito,
    mientras que aprobar a un moroso cuesta la porcion no recuperada del
    capital, y el balanceo con datos sinteticos desplaza las probabilidades
    al alza.

    El costo se calcula con el capital real de cada registro, porque rechazar
    un credito de cinco millones no equivale a rechazar uno de quinientos mil.

    Args:
        pipeline: modelo ajustado.
        X: predictoras sin codificar.
        y: objetivo observado.
        config: configuracion del proyecto.
        pasos: cantidad de umbrales a evaluar entre 0,05 y 0,95.

    Returns:
        DataFrame con una fila por umbral evaluado.
    """
    costos = config["decision"]["costos"]
    margen = costos["margen_sobre_capital"]
    severidad = costos["severidad_perdida"]

    capital = np.expm1(X["capital_prestado"]) if X["capital_prestado"].max() < 30 \
        else X["capital_prestado"]

    incumplio = (y == INCUMPLE).to_numpy()
    probabilidades = pipeline.predict_proba(X)[:, 1]

    filas = []
    for umbral in np.linspace(0.05, 0.95, pasos):
        rechazado = probabilidades >= umbral

        buenos_rechazados = rechazado & ~incumplio
        morosos_aprobados = ~rechazado & incumplio
        morosos_detectados = rechazado & incumplio

        costo_oportunidad = (capital[buenos_rechazados] * margen).sum()
        costo_perdida = (capital[morosos_aprobados] * severidad).sum()

        filas.append({
            "umbral": round(umbral, 3),
            "tasa_rechazo": round(rechazado.mean(), 4),
            "morosos_detectados": int(morosos_detectados.sum()),
            "buenos_rechazados": int(buenos_rechazados.sum()),
            "precision": round(
                morosos_detectados.sum() / rechazado.sum(), 4)
            if rechazado.sum() else 0.0,
            "recall": round(morosos_detectados.sum() / incumplio.sum(), 4),
            "costo_oportunidad": round(costo_oportunidad),
            "costo_perdida": round(costo_perdida),
            "costo_total": round(costo_oportunidad + costo_perdida),
        })

    return pd.DataFrame(filas)


def seleccionar_umbral(tabla: pd.DataFrame,
                       config: dict[str, Any]) -> dict[str, Any]:
    """Elige el umbral combinando el criterio de costo y el operativo.

    El criterio de costo busca el minimo de la curva de costo total. El
    criterio operativo identifica los umbrales que producen tasas de rechazo
    manejables para revision manual. Cuando ambos coinciden la decision es
    solida; cuando divergen, el optimo de costo exige un volumen de rechazo
    que la operacion no absorbe y hay que decidir explicitamente.

    Returns:
        Diccionario con el umbral de costo minimo y los umbrales operativos.
    """
    fila_costo = tabla.loc[tabla["costo_total"].idxmin()]

    operativos = {}
    for objetivo in config["decision"]["tasas_rechazo_objetivo"]:
        indice = (tabla["tasa_rechazo"] - objetivo).abs().idxmin()
        operativos[objetivo] = tabla.loc[indice]

    logger.info("--- Analisis del umbral de corte ---")
    logger.info("\n%s", tabla.to_string(index=False))

    logger.info("Criterio de costo minimo: umbral %.3f", fila_costo["umbral"])
    logger.info(
        "    Rechaza %.1f%% de la cartera | detecta %d de %d morosos | "
        "precision %.4f | costo total $%s",
        fila_costo["tasa_rechazo"] * 100,
        fila_costo["morosos_detectados"],
        tabla["morosos_detectados"].max(),
        fila_costo["precision"],
        f"{int(fila_costo['costo_total']):,}",
    )

    logger.info("Criterio operativo:")
    for objetivo, fila in operativos.items():
        logger.info(
            "    Rechazo objetivo %.0f%% -> umbral %.3f (rechazo real %.1f%%, "
            "precision %.4f, detecta %d morosos)",
            objetivo * 100, fila["umbral"], fila["tasa_rechazo"] * 100,
            fila["precision"], fila["morosos_detectados"],
        )

    return {"costo": fila_costo, "operativos": operativos}


def analisis_sensibilidad(pipeline, X: pd.DataFrame, y: pd.Series,
                          config: dict[str, Any]) -> pd.DataFrame:
    """Verifica si el umbral optimo depende de los supuestos de costo.

    Los valores de margen y severidad son supuestos razonables, no datos de
    la entidad. Si el umbral resultante se mantiene estable ante variaciones
    amplias de esos supuestos, la decision es robusta; si cambia mucho,
    conviene declarar que el resultado depende de parametros no verificados.
    """
    escenarios = [
        ("conservador", 0.15, 0.50),
        ("base", config["decision"]["costos"]["margen_sobre_capital"],
         config["decision"]["costos"]["severidad_perdida"]),
        ("agresivo", 0.06, 0.80),
    ]

    filas = []
    for nombre, margen, severidad in escenarios:
        config_escenario = {
            "decision": {
                "costos": {"margen_sobre_capital": margen,
                           "severidad_perdida": severidad},
                "tasas_rechazo_objetivo": [],
            }
        }
        tabla = analizar_umbrales(pipeline, X, y, config_escenario)
        mejor = tabla.loc[tabla["costo_total"].idxmin()]
        filas.append({
            "escenario": nombre,
            "margen": margen,
            "severidad": severidad,
            "ratio_costos": round(severidad / margen, 1),
            "umbral_optimo": mejor["umbral"],
            "tasa_rechazo": mejor["tasa_rechazo"],
        })

    resultado = pd.DataFrame(filas)
    logger.info("--- Sensibilidad del umbral a los supuestos de costo ---")
    logger.info("\n%s", resultado.to_string(index=False))
    return resultado

# ============================================================================
# EJECUCION
# ============================================================================
def main() -> None:
    """Ejecuta el proceso completo de entrenamiento y evaluacion."""
    config = cargar_config()
    conjuntos = cargar_conjuntos(config)
    X_train, y_train = conjuntos["X_train"], conjuntos["y_train"]
    X_test, y_test = conjuntos["X_test"], conjuntos["y_test"]

    modelos = definir_modelos(config)
    mediciones = validar_modelos(modelos, X_train, y_train, config)
    resumen = resumir_comparacion(mediciones)

    contraste = comparar_estadisticamente(mediciones)
    finalistas = seleccionar_finalistas(resumen, contraste)

    optimizados = optimizar_finalistas(
        finalistas, modelos, X_train, y_train, config)

    ganador = seleccionar_ganador(optimizados, resumen)
    logger.info("Modelo ganador: %s optimizado con %s (PR-AUC interno %.4f)",
                ganador["modelo"], ganador["metodo"], ganador["pr_auc"])

    metricas_test = evaluar_en_test(
        ganador["pipeline"], X_test, y_test,
        nombre=f"{ganador['modelo']} ({ganador['metodo']})")

    metricas_heuristico = evaluar_heuristico_como_linea_base(
        X_test, y_test, config)

    raiz = Path(__file__).resolve().parents[2]

    tabla_umbrales = analizar_umbrales(
        ganador["pipeline"], X_train, y_train, config)
    seleccion = seleccionar_umbral(tabla_umbrales, config)
    analisis_sensibilidad(ganador["pipeline"], X_train, y_train, config)

    comparacion_final = pd.DataFrame({
        "heuristico": metricas_heuristico,
        ganador["modelo"]: metricas_test,
    }).round(4)
    logger.info("--- Aporte del aprendizaje automatico sobre las reglas ---")
    logger.info("\n%s", comparacion_final.to_string())

    
    generar_graficos(mediciones, resumen, ganador["pipeline"],
                     X_test, y_test, raiz / "mlops_pipeline" / "reportes")

    from ft_engineering import construir_pipeline_limpieza

    pipeline_limpieza = construir_pipeline_limpieza(config)
    pipeline_limpieza.fit_transform(
        pd.read_csv(
            raiz / config["datos"]["ruta"], sep=config["datos"]["separador"]
        ).drop(columns=[config["datos"]["columna_objetivo"]], errors="ignore")
    )

    guardar_modelo(
        ganador["pipeline"],
        metadatos={
            "modelo": ganador["modelo"],
            "metodo_optimizacion": ganador["metodo"],
            "parametros": ganador["parametros"],
            "pr_auc_validacion": ganador["pr_auc"],
            "metricas_test": metricas_test,
            "features": list(X_train.columns),
            "pipeline_limpieza": pipeline_limpieza,
            "nota": "El artefacto incluye el pipeline de limpieza ajustado, "
                    "de modo que el despliegue no requiere acceso al dataset "
                    "historico.",
        },
        ruta=raiz / "mlops_pipeline" / "modelo_final.joblib",
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()