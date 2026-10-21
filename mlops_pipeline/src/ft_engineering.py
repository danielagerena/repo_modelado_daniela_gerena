"""
ft_engineering.py

Componente de feature engineering del pipeline de modelado de riesgo
crediticio. Lee la configuracion desde config.json, aplica limpieza,
reglas de calidad y variables derivadas, y retorna los conjuntos de
entrenamiento y evaluacion.

Toda la parametrizacion vive en mlops_pipeline/config.json. Este modulo
contiene unicamente la logica de transformacion.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)

# ft_engineering.py esta en <raiz>/mlops_pipeline/src/, por lo que la raiz
# del proyecto queda dos niveles arriba.
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_PROYECTO / "mlops_pipeline" / "config.json"


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict[str, Any]:
    """Lee el archivo de configuracion del pipeline.

    Args:
        ruta: ubicacion del config.json. Por defecto, el del proyecto.

    Returns:
        Diccionario con la configuracion.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el contenido no es JSON valido.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {ruta}")

    try:
        with ruta.open(encoding="utf-8") as archivo:
            config = json.load(archivo)
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo {ruta} no es un JSON valido: {error}") from error

    logger.info("Configuracion cargada desde %s", ruta)
    return config


def _leer_csv(parametros: dict[str, Any]) -> pd.DataFrame:
    """Lee el dataset desde el archivo plano definido en el config.

    Args:
        parametros: seccion "datos" del config.

    Returns:
        DataFrame con los datos sin transformar.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    ruta_datos = RAIZ_PROYECTO / parametros["ruta"]

    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {ruta_datos}")

    return pd.read_csv(ruta_datos, sep=parametros["separador"])


def _normalizar_tipos_bigquery(datos: pd.DataFrame) -> pd.DataFrame:
    """Deja los tipos como los entregaria pd.read_csv.

    BigQuery entrega los enteros como Int64 (entero con nulos de pandas) y las
    fechas con zona horaria. Los enteros pasan a int64 si no tienen nulos y a
    float64 si los tienen; las fechas pierden la zona horaria. Asi el resto
    del pipeline recibe los mismos tipos que con el archivo plano.

    Args:
        datos: DataFrame tal como lo entrega BigQuery.

    Returns:
        DataFrame con tipos equivalentes a los de la lectura desde CSV.
    """
    datos = datos.copy()
    for columna in datos.select_dtypes(include=["Int64"]).columns:
        tipo = "int64" if datos[columna].notna().all() else "float64"
        datos[columna] = datos[columna].astype(tipo)
    for columna in datos.select_dtypes(include=["datetimetz"]).columns:
        datos[columna] = datos[columna].dt.tz_localize(None)
    return datos


def _leer_bigquery(parametros_bq: dict[str, Any]) -> pd.DataFrame:
    """Lee el dataset desde la tabla de BigQuery definida en el config.

    La autenticacion usa las credenciales predeterminadas de aplicacion
    (gcloud auth application-default login), sin llaves de cuenta de servicio.
    Las filas se ordenan por su contenido completo porque BigQuery no
    garantiza el orden de un SELECT *, y la particion train/test depende de
    ese orden: sin ordenar, cada ejecucion podria dar metricas distintas.

    Args:
        parametros_bq: subseccion "datos.bigquery" del config.

    Returns:
        DataFrame con los datos sin transformar.

    Raises:
        ImportError: si las librerias de BigQuery no estan instaladas.
    """
    try:
        from google.cloud import bigquery
    except ImportError as error:
        raise ImportError(
            "Para leer desde BigQuery instale google-cloud-bigquery, db-dtypes y pyarrow"
        ) from error

    tabla = (
        f"{parametros_bq['project_id']}.{parametros_bq['dataset']}."
        f"{parametros_bq['tabla']}"
    )
    consulta = f"SELECT * FROM `{tabla}` AS t ORDER BY TO_JSON_STRING(t)"

    logger.info("Consultando BigQuery: %s", tabla)
    cliente = bigquery.Client(project=parametros_bq["project_id"])
    datos = cliente.query(consulta).to_dataframe()
    return _normalizar_tipos_bigquery(datos)


def cargar_datos(config: dict[str, Any]) -> pd.DataFrame:
    """Carga el dataset crudo desde la fuente definida en el config.

    La fuente se elige con config["datos"]["fuente"]: "bigquery" lee la tabla
    configurada en GCP y "csv" lee el archivo plano. Si la clave no existe se
    usa "csv", para mantener compatibilidad con configuraciones anteriores.

    Args:
        config: configuracion del pipeline.

    Returns:
        DataFrame con los datos sin transformar.

    Raises:
        FileNotFoundError: si la fuente es csv y el archivo no existe.
        ValueError: si la fuente no es reconocida.
    """
    parametros = config["datos"]
    fuente = parametros.get("fuente", "csv")

    if fuente == "bigquery":
        datos = _leer_bigquery(parametros["bigquery"])
    elif fuente == "csv":
        datos = _leer_csv(parametros)
    else:
        raise ValueError(
            f"Fuente de datos no reconocida: {fuente}. Use 'bigquery' o 'csv'."
        )

    logger.info("Dataset cargado desde %s: %d filas, %d columnas", fuente, *datos.shape)
    return datos

    # ============================================================================
# TRANSFORMADORES DE LIMPIEZA Y CALIDAD
# ============================================================================
class CastTypes(BaseEstimator, TransformerMixin):
    """Corrige tipos mal cargados y unifica la representacion de nulos.

    Los valores fuera del dominio declarado de una categorica se marcan como
    nulos (por ejemplo, valores numericos infiltrados en tendencia_ingresos).
    """

    def __init__(self, decimal_coma=None, a_datetime=None, categorias_validas=None):
        self.decimal_coma = decimal_coma
        self.a_datetime = a_datetime
        self.categorias_validas = categorias_validas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for columna in self.decimal_coma or []:
            if columna in X.columns:
                X[columna] = pd.to_numeric(
                    X[columna].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
        for columna in self.a_datetime or []:
            if columna in X.columns:
                X[columna] = pd.to_datetime(X[columna], dayfirst=True, errors="coerce")
        for columna, validas in (self.categorias_validas or {}).items():
            if columna in X.columns:
                valores = X[columna].astype(str).str.strip()
                X[columna] = valores.where(valores.isin(validas), np.nan)
        return X


class DropDuplicates(BaseEstimator, TransformerMixin):
    """Elimina registros duplicados (paso 1 de la preparacion de datos)."""

    def __init__(self, activo=True):
        self.activo = activo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.activo:
            return X
        resultado = X.drop_duplicates().copy()
        eliminados = len(X) - len(resultado)
        if eliminados:
            logger.info("Duplicados eliminados: %d", eliminados)
        return resultado


class DropColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas por nombre, ignorando las que no existan."""

    def __init__(self, columnas=None):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columnas or [], errors="ignore")


class FiltrarCategorias(BaseEstimator, TransformerMixin):
    """Restringe el alcance del modelo a las categorias con muestra suficiente.

    Elimina filas, no columnas: las categorias descartadas quedan fuera del
    alcance declarado del modelo en lugar de imputarse.
    """

    def __init__(self, filtros=None):
        self.filtros = filtros

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for columna, valores in (self.filtros or {}).items():
            if columna in X.columns:
                antes = len(X)
                X = X[X[columna].isin(valores)]
                logger.info(
                    "Filtro sobre %s: %d filas eliminadas", columna, antes - len(X)
                )
        return X.copy()


class SentinelToNull(BaseEstimator, TransformerMixin):
    """Convierte en nulos los valores centinela.

    Un centinela es valido en el tipo de dato pero invalido en el negocio
    (un salario de 0 no es un ingreso, es un dato ausente).
    """

    def __init__(self, centinelas=None):
        self.centinelas = centinelas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for columna, valor in (self.centinelas or {}).items():
            if columna in X.columns:
                X[columna] = X[columna].replace(valor, np.nan)
        return X


class ImputacionCruzadaIngresos(BaseEstimator, TransformerMixin):
    """Recupera el salario ausente usando el ingreso reportado por la central.

    Ambas variables existen antes de la decision crediticia, por lo que el
    cruce no introduce fuga de informacion. Se ejecuta antes del filtro por
    porcentaje de nulos para aprovechar la variable de origen justo antes de
    que ese filtro la elimine.
    """

    def __init__(self, destino="salario_cliente",
                 origen="promedio_ingresos_datacredito"):
        self.destino = destino
        self.origen = origen

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.destino in X.columns and self.origen in X.columns:
            recuperables = X[self.destino].isna() & X[self.origen].notna()
            X.loc[recuperables, self.destino] = X.loc[recuperables, self.origen]
            logger.info("Salarios recuperados de la central: %d", int(recuperables.sum()))
        return X


class DropByNullRate(BaseEstimator, TransformerMixin):
    """Elimina columnas cuyo porcentaje de nulos supere el umbral.

    Por encima del umbral la imputacion deja de completar informacion y pasa
    a fabricar la distribucion de la variable. imputarlos introduce sesgo.

    El umbral se aprende en fit para que la decision se tome sobre los datos
    de entrenamiento y no sobre el conjunto completo.
    """

    def __init__(self, umbral=0.20, excluir=None):
        self.umbral = umbral
        self.excluir = excluir

    def fit(self, X, y=None):
        excluir = self.excluir or []
        tasa_nulos = X.isna().mean()
        self.columnas_eliminadas_ = [
            columna
            for columna in tasa_nulos[tasa_nulos > self.umbral].index
            if columna not in excluir
        ]
        if self.columnas_eliminadas_:
            logger.info(
                "Columnas con mas de %.0f%% de nulos eliminadas: %s",
                self.umbral * 100,
                self.columnas_eliminadas_,
            )
        return self

    def transform(self, X):
        return X.drop(columns=self.columnas_eliminadas_, errors="ignore")


class FlagOutliersAsNull(BaseEstimator, TransformerMixin):
    """Marca como nulos los valores que incumplen las reglas de calidad.

    Las reglas provienen del contexto crediticio colombiano, no de los rangos
    observados en el dataset. El registro no se elimina: el valor invalido se
    convierte en nulo y se imputa mas adelante.
    """

    def __init__(self, reglas=None):
        self.reglas = reglas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for columna, (minimo, maximo) in (self.reglas or {}).items():
            if columna not in X.columns:
                continue
            fuera_de_rango = pd.Series(False, index=X.index)
            if minimo is not None:
                fuera_de_rango |= X[columna] < minimo
            if maximo is not None:
                fuera_de_rango |= X[columna] > maximo
            if fuera_de_rango.any():
                logger.info(
                    "Regla de calidad en %s: %d valores marcados como nulos",
                    columna,
                    int(fuera_de_rango.sum()),
                )
            X.loc[fuera_de_rango, columna] = np.nan
        return X


class ReglasCoherencia(BaseEstimator, TransformerMixin):
    """Reglas de calidad que involucran mas de una columna.

    Igual que las reglas de rango, marcan nulos en lugar de eliminar filas.
    """

    def __init__(self, smmlv=None):
        self.smmlv = smmlv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Una cuota superior al ingreso solo se considera error de registro
        # cuando el ingreso mismo es invalido. Con un ingreso valido, una cuota
        # que lo supera no es un dato mal capturado sino el caso extremo de
        # sobreendeudamiento, y debe conservarse para que las reglas de
        # negocio puedan evaluarlo.
        if {"cuota_pactada", "salario_cliente"} <= set(X.columns):
            ingreso_invalido = X["salario_cliente"].isna() | (X["salario_cliente"] <= 0)
            incoherente = (
                (X["cuota_pactada"] > X["salario_cliente"]) & ingreso_invalido
            ).fillna(False)
            X.loc[incoherente, "cuota_pactada"] = np.nan

        # La mora es una porcion del saldo y no puede excederlo.
        if {"saldo_mora", "saldo_total"} <= set(X.columns):
            incoherente = (X["saldo_mora"] > X["saldo_total"]).fillna(False)
            X.loc[incoherente, "saldo_mora"] = np.nan

        # A un empleado no se le puede pagar menos del minimo legal, de modo
        # que un salario inferior es un dato invalido. Se anula solo cuando la
        # cuota tambien resulta coherente con ese ingreso: si la cuota lo
        # supera ampliamente, anular el salario impediria que las reglas de
        # negocio evaluen el sobreendeudamiento, que es justamente el caso que
        # deben capturar.
        if self.smmlv and {"salario_cliente", "tipo_laboral"} <= set(X.columns):
            bajo_minimo = (
                (X["tipo_laboral"] == "Empleado") & (X["salario_cliente"] < self.smmlv)
            ).fillna(False)
            if "cuota_pactada" in X.columns:
                cuota_excede = (X["cuota_pactada"] > X["salario_cliente"]).fillna(False)
                bajo_minimo &= ~cuota_excede
            X.loc[bajo_minimo, "salario_cliente"] = np.nan

        return X

# ============================================================================
# TRANSFORMACIONES DE NEGOCIO
# ============================================================================
class NuevasVariables(BaseEstimator, TransformerMixin):
    """Crea las variables derivadas con significado financiero.

    Todas se calculan con informacion disponible en el momento de decidir el
    credito, por lo que ninguna introduce fuga de informacion.

    esfuerzo_pago
        cuota / ingreso. Capacidad de endeudamiento del cliente. Es el ratio
        cuota-ingreso del analisis de capacidad de pago del SARC.
    saldo_relativo_ingreso
        saldo reportado en central / ingreso. Peso de las obligaciones
        vigentes frente a la capacidad mensual.
    carga_deuda_previa
        otros prestamos / ingreso. Apalancamiento acumulado antes de este
        credito.
    intensidad_busqueda_credito
        consultas por credito vigente. El EDA mostro mas huella de consulta
        entre los incumplidores; el ratio distingue la actividad normal de un
        cliente bancarizado de la busqueda urgente de liquidez.
    tiene_mora_reportada
        indicador binario de mora en central. El monto no aporta escala
        (99.5% en cero) pero la condicion es la base de una regla dura.
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _ratio(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
        """Divide evitando la division por cero, que se convierte en nulo."""
        return numerador / denominador.replace(0, np.nan)

    def transform(self, X):
        X = X.copy()
        ingreso = X["salario_cliente"]

        X["esfuerzo_pago"] = self._ratio(X["cuota_pactada"], ingreso)
        X["saldo_relativo_ingreso"] = self._ratio(X["saldo_total"], ingreso)
        X["carga_deuda_previa"] = self._ratio(X["total_otros_prestamos"], ingreso)
        X["intensidad_busqueda_credito"] = X["huella_consulta"] / (
            X["cant_creditosvigentes"] + 1
        )
        X["tiene_mora_reportada"] = (X["saldo_mora"] > 0).astype(int)

        logger.info("Variables derivadas creadas: %d", 5)
        return X


class DiscretizarScore(BaseEstimator, TransformerMixin):
    """Convierte el puntaje de central de riesgo en una variable ordinal.

    Los cortes provienen de la escala publica de Datacredito-Experian y de la
    lectura de mercado sobre ella, no de los cuartiles del dataset. La central
    no publica tramos oficiales, por lo que se documenta la fuente en el
    archivo de configuracion.
    """

    def __init__(self, origen=None, destino=None, cortes=None, etiquetas=None):
        self.origen = origen
        self.destino = destino
        self.cortes = cortes
        self.etiquetas = etiquetas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.origen not in X.columns:
            return X

        X[self.destino] = pd.cut(
            X[self.origen],
            bins=self.cortes,
            labels=self.etiquetas,
            right=False,
            include_lowest=True,
        ).astype(object)

        sin_clasificar = X[self.destino].isna().sum() - X[self.origen].isna().sum()
        if sin_clasificar > 0:
            logger.warning(
                "%d valores de %s quedaron fuera de los tramos definidos",
                sin_clasificar,
                self.origen,
            )
        return X

# ============================================================================
# ENSAMBLADO DE LOS PIPELINES
# ============================================================================
def construir_pipeline_limpieza(config: dict[str, Any]) -> Pipeline:
    """Arma el pipeline de limpieza y creacion de variables.

    Sigue el orden de la metodologia de preparacion de datos: integracion,
    eliminacion de irrelevantes y fugas, alcance, calidad, creacion de nuevas
    variables y eliminacion de redundantes.

    La imputacion NO ocurre aqui: vive dentro del ColumnTransformer para que
    los estadisticos se aprendan solo sobre el conjunto de entrenamiento.

    Args:
        config: configuracion del pipeline.

    Returns:
        Pipeline que recibe el DataFrame crudo y devuelve el DataFrame con
        las variables listas para codificar, incluida la columna objetivo.
    """
    integracion = config["integracion"]
    eliminacion = config["eliminacion"]
    calidad = config["calidad"]
    discretizacion = config["discretizacion"]["rango_datacredito"]

    return Pipeline([
        ("tipos", CastTypes(
            decimal_coma=integracion["decimal_coma"],
            a_datetime=integracion["a_datetime"],
            categorias_validas=integracion["categorias_validas"],
        )),
        ("duplicados", DropDuplicates(activo=integracion["eliminar_duplicados"])),
        ("descartes", DropColumns(
            columnas=eliminacion["leakage"] + eliminacion["irrelevantes"],
        )),
        ("alcance", FiltrarCategorias(filtros=config["alcance"]["filtro_categorias"])),
        ("centinelas", SentinelToNull(centinelas=calidad["centinelas_a_nulo"])),
        ("ingresos_cruzados", ImputacionCruzadaIngresos()),
        ("filtro_nulos", DropByNullRate(
            umbral=eliminacion["umbral_nulos"],
            excluir=[config["datos"]["columna_objetivo"]],
        )),
        ("reglas_rango", FlagOutliersAsNull(reglas=calidad["reglas_rango"])),
        ("reglas_coherencia", ReglasCoherencia(
            smmlv=config["contexto_colombia"]["smmlv"],
        )),
        ("nuevas_variables", NuevasVariables()),
        ("discretizar_score", DiscretizarScore(
            origen=discretizacion["origen"],
            destino="rango_datacredito",
            cortes=discretizacion["cortes"],
            etiquetas=discretizacion["etiquetas"],
        )),
        ("redundantes", DropColumns(columnas=eliminacion["redundantes"])),
    ])


def construir_preprocesador(config: dict[str, Any]) -> ColumnTransformer:
    """Arma el ColumnTransformer con una rama por tipo de variable.

    Cada rama imputa sus propios nulos antes de codificar, de modo que los
    estadisticos de imputacion se aprenden unicamente sobre entrenamiento.

        numericas             SimpleImputer(mediana) + StandardScaler
        categoricas nominales SimpleImputer(moda) + OneHotEncoder
        categoricas ordinales SimpleImputer(moda) + OrdinalEncoder
        binarias              sin transformacion

    Args:
        config: configuracion del pipeline.

    Returns:
        ColumnTransformer sin ajustar.
    """
    features = config["features"]
    ordinales = features["categoricas_ordinales"]

    rama_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])

    rama_numerica_log = Pipeline([
        ("logaritmo", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])

    rama_nominal = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    rama_ordinal = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OrdinalEncoder(
            categories=[ordinales[columna] for columna in ordinales],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("escalar", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numericas", rama_numerica, features["numericas"]),
            ("numericas_log", rama_numerica_log, features["numericas_log"]),
            ("nominales", rama_nominal, features["categoricas_nominales"]),
            ("ordinales", rama_ordinal, list(ordinales)),
            ("binarias", "passthrough", features["binarias"]),
        ],
        remainder="drop",
    )

# ============================================================================
# DIAGNOSTICO DE CORRELACIONES
# ============================================================================
def _asociacion_con_objetivo(serie: pd.Series, objetivo: pd.Series) -> float:
    """Mide la asociacion de una variable con el objetivo binario.

    Usa Spearman para variables numericas y el ratio de correlacion (eta)
    para categoricas. Ambas medidas quedan en [0, 1], por lo que son
    comparables entre si.
    """
    valida = serie.notna() & objetivo.notna()
    if valida.sum() < 3:
        return np.nan

    if pd.api.types.is_numeric_dtype(serie):
        correlacion = serie[valida].corr(objetivo[valida], method="spearman")
        return abs(correlacion) if pd.notna(correlacion) else np.nan

    categoria, numero = serie[valida], objetivo[valida]
    if categoria.nunique() < 2:
        return np.nan
    media = numero.mean()
    entre = sum(
        len(grupo) * (grupo.mean() - media) ** 2
        for _, grupo in numero.groupby(categoria, observed=True)
    )
    total = ((numero - media) ** 2).sum()
    return np.sqrt(entre / total) if total > 0 else np.nan


def diagnosticar_correlaciones(datos: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Reporta redundancia entre numericas y relevancia frente al objetivo.

    No modifica los datos: la eliminacion de variables es una decision
    documentada que se registra en el config, no un efecto automatico.

    Args:
        datos: DataFrame limpio, incluida la columna objetivo.
        config: configuracion del pipeline.

    Returns:
        DataFrame con la asociacion de cada variable con el objetivo,
        ordenado de mayor a menor.
    """
    parametros = config["correlaciones"]
    objetivo_col = config["datos"]["columna_objetivo"]
    objetivo = pd.to_numeric(datos[objetivo_col], errors="coerce")
    predictoras = datos.drop(columns=[objetivo_col])

    numericas = predictoras.select_dtypes(include=np.number)

    relevancia = pd.Series(
        {columna: _asociacion_con_objetivo(predictoras[columna], objetivo)
         for columna in predictoras.columns},
        name="asociacion_objetivo",
    ).sort_values(ascending=False)

    logger.info("Asociacion de cada variable con el objetivo:")
    for columna, valor in relevancia.items():
        logger.info("    %-32s %.4f", columna, valor)

    matriz = numericas.corr(method="spearman").abs()
    umbral = parametros["umbral_redundancia"]

    for i, primera in enumerate(matriz.columns):
        for segunda in matriz.columns[i + 1:]:
            if matriz.loc[primera, segunda] < umbral:
                continue
            relevancia_primera = relevancia.get(primera, np.nan)
            relevancia_segunda = relevancia.get(segunda, np.nan)
            conservar = primera if relevancia_primera >= relevancia_segunda else segunda
            logger.warning(
                "Redundancia %.2f entre %s (%.4f) y %s (%.4f). Conservar: %s",
                matriz.loc[primera, segunda],
                primera, relevancia_primera,
                segunda, relevancia_segunda,
                conservar,
            )

    debiles = relevancia[relevancia < parametros["umbral_irrelevancia"]].index.tolist()
    if debiles:
        logger.info(
            "Variables por debajo del umbral de relevancia (revisar, no eliminar "
            "automaticamente): %s", debiles,
        )

    return relevancia.to_frame()


# ============================================================================
# PARTICION Y BALANCEO
# ============================================================================
def balancear(X_train, y_train, config: dict[str, Any]):
    """Balancea la clase minoritaria unicamente sobre entrenamiento.

    Aplicarlo antes de la particion haria que registros sinteticos derivados
    de filas de evaluacion terminaran en entrenamiento. El conjunto de
    evaluacion conserva la prevalencia real porque es la que vera el modelo
    en produccion.

    Las reglas del modelo heuristico se calibran sobre el conjunto sin
    balancear: un umbral ajustado sobre datos donde la clase minoritaria fue
    elevada al 50% sobreestima su precision.

    Raises:
        ValueError: si el metodo configurado no esta soportado.
        ImportError: si se pide SMOTE sin imbalanced-learn instalado.
    """
    metodo = config["balanceo"]["metodo"]
    semilla = config["particion"]["random_state"]

    if metodo != "smote":
        raise ValueError(f"Metodo de balanceo no soportado: {metodo}")

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as error:
        raise ImportError(
            "El balanceo con SMOTE requiere imbalanced-learn. "
            "Instalalo con: pip install imbalanced-learn"
        ) from error

    X_balanceado, y_balanceado = SMOTE(
        sampling_strategy=config["balanceo"].get("sampling_strategy", 1.0),
        random_state=semilla,
    ).fit_resample(X_train, y_train)
    logger.info("Entrenamiento balanceado: %d filas", len(X_balanceado))
    return X_balanceado, y_balanceado


def generar_datasets(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta el feature engineering completo y retorna los conjuntos.

    Es la funcion que consume model_training.py. No escribe archivos: los
    datos se pasan en memoria para evitar que el entrenamiento use un
    resultado desactualizado del pipeline.

    Returns:
        Diccionario con X_train, X_test, y_train, y_test, las versiones
        balanceadas de entrenamiento, el preprocesador ya ajustado y el
        DataFrame limpio sin codificar (base para el modelo heuristico).
    """
    config = config or cargar_config()
    crudos = cargar_datos(config)

    limpios = construir_pipeline_limpieza(config).fit_transform(crudos)
    logger.info("Datos limpios: %d filas, %d columnas", *limpios.shape)


    diagnosticar_correlaciones(limpios, config)

    objetivo_col = config["datos"]["columna_objetivo"]
    y = pd.to_numeric(limpios[objetivo_col]).astype(int)
    X = limpios.drop(columns=[objetivo_col])

    declaradas = (
        config["features"]["numericas"]
        + config["features"]["numericas_log"]
        + config["features"]["categoricas_nominales"]
        + list(config["features"]["categoricas_ordinales"])
        + config["features"]["binarias"]
    )
    faltantes = [columna for columna in declaradas if columna not in X.columns]
    if faltantes:
        raise ValueError(f"Features declaradas en config que no existen: {faltantes}")

    particion = config["particion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=particion["test_size"],
        random_state=particion["random_state"],
        stratify=y if particion["estratificar"] else None,
    )

    preprocesador = construir_preprocesador(config)
    X_train_cod = pd.DataFrame(
        preprocesador.fit_transform(X_train),
        columns=preprocesador.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_cod = pd.DataFrame(
        preprocesador.transform(X_test),
        columns=preprocesador.get_feature_names_out(),
        index=X_test.index,
    )
    logger.info(
        "Entrenamiento: %d filas | Evaluacion: %d filas (sin balancear)",
        len(X_train_cod), len(X_test_cod),
    )

    X_train_bal, y_train_bal = balancear(X_train_cod, y_train, config)

    return {
        "X_train": X_train_cod,
        "X_test": X_test_cod,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_balanceado": X_train_bal,
        "y_train_balanceado": y_train_bal,
        "preprocesador": preprocesador,
        "datos_limpios": limpios,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    resultado = generar_datasets()
    logger.info("Features generadas: %d", resultado["X_train"].shape[1])
    logger.info(
        "Tasa de incumplimiento en evaluacion: %.4f", 1 - resultado["y_test"].mean(),
    )