# Modelo de riesgo crediticio — cartera de consumo (Colombia)

Pipeline de MLOps para predecir el incumplimiento de pago en obligaciones de
crédito de consumo, a partir de información disponible en el momento de la
decisión crediticia.

**Objetivo:** `Pago_atiempo` — 1 si la obligación fue pagada en su totalidad,
0 si no. La variable se registra una vez cerrada la obligación, ya sea por
vencimiento del plazo o por prepago del cliente.

**Autora:** Daniela Gerena
**Curso:** Ciencia de Datos en Producción — Universidad Pontificia Bolivariana

---

## Arquitectura

```
Datos → Feature Engineering → Reglas heurísticas → Modelo ML → Decisión combinada
                                       ↓                            ↓
                              (prevalecen siempre)          Monitoreo / datadrift
```

Las reglas duras se evalúan primero y prevalecen sobre cualquier predicción
estadística. Solo las solicitudes que ninguna regla marca llegan al
clasificador. Esta separación permite que cada rechazo tenga un motivo
auditable y que la política de crédito no dependa de la disponibilidad del
modelo.

---

## Estructura del repositorio

```
repo_modelado_daniela_gerena/
├── mlops_pipeline/
│   ├── src/
│   │   ├── Cargar_datos.ipynb            Materialización de la fuente
│   │   ├── comprension_eda.ipynb         Análisis exploratorio
│   │   ├── ft_engineering.py             Limpieza, features y partición
│   │   ├── hueristic_model.py            Reglas duras de política de crédito
│   │   ├── model_training_evaluation.py  Entrenamiento, comparación y evaluación
│   │   ├── model_deploy.py               Lógica de decisión combinada
│   │   ├── app.py                        Interfaz Streamlit
│   │   └── model_monitoring.py           Registro y detección de datadrift
│   ├── config.json                       Parametrización del pipeline
│   └── modelo_final.joblib               Artefacto desplegable
├── Base_de_datos.csv                     Dataset (10.763 registros)
├── sonar-project.properties
├── requirements.txt
├── set_up.bat
└── readme.md
```

**Nota sobre la estructura:** el docente solicitó unificar `model_training` y
`model_evaluation` en un solo componente, de ahí el nombre
`model_training_evaluation.py`. `app.py` no figura en el lineamiento original;
se añadió porque el despliegue se resolvió con Streamlit en lugar de una API
independiente.

---

## Instalación

```bat
set_up.bat
```

Crea el entorno virtual e instala las dependencias. Luego:

```powershell
.venv\Scripts\activate

python mlops_pipeline\src\ft_engineering.py
python mlops_pipeline\src\hueristic_model.py
python mlops_pipeline\src\model_training_evaluation.py
python mlops_pipeline\src\model_deploy.py
python mlops_pipeline\src\model_monitoring.py

streamlit run mlops_pipeline\src\app.py
```

Toda la parametrización vive en `mlops_pipeline/config.json`: rutas, umbrales,
reglas de calidad, reglas de negocio y listas de variables. El código no
contiene valores fijos.

---

## Diccionario de datos

El dataset original tiene 10.763 registros y 23 variables. El diccionario se
construyó a partir del análisis exploratorio y del funcionamiento del sistema
de crédito colombiano; el proveedor no entregó documentación de origen. Las
interpretaciones marcadas con (*) son inferencias validadas contra el
comportamiento de los datos.

### Variables del crédito

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `tipo_credito` | Categórica | Línea o producto. Códigos 4 (72%) y 9 (27%) concentran el 98,7% | Feature nominal |
| `fecha_prestamo` | Fecha | Desembolso. Rango 26/11/2024 – 26/04/2026 | Eliminada |
| `capital_prestado` | Numérica | Monto desembolsado, en pesos | Feature numérica |
| `plazo_meses` | Numérica | Plazo pactado en meses | Feature numérica |
| `cuota_pactada` | Numérica | Valor de la cuota periódica | Base de `esfuerzo_pago` |

### Variables del cliente

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `edad_cliente` | Numérica | Edad al momento del desembolso | Feature numérica |
| `tipo_laboral` | Categórica | Empleado o Independiente | Feature nominal |
| `salario_cliente` | Numérica | Ingreso mensual declarado | Feature numérica |
| `promedio_ingresos_datacredito` | Numérica | Ingreso promedio reportado por la central (*) | Eliminada por nulos |
| `tendencia_ingresos` | Categórica | Creciente, Decreciente, Estable | Eliminada por nulos |

### Variables de comportamiento crediticio

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `puntaje` | Numérica | Score interno de la entidad | **Eliminada (fuga)** |
| `puntaje_datacredito` | Numérica | Score de central, escala 150–950 | Base de `rango_datacredito` |
| `cant_creditosvigentes` | Numérica | Obligaciones vigentes | Base de `intensidad_busqueda_credito` |
| `huella_consulta` | Numérica | Consultas recientes en centrales | Feature numérica |
| `saldo_total` | Numérica | Saldo reportado, con intereses y cargos (*) | Base de `saldo_relativo_ingreso` |
| `saldo_principal` | Numérica | Saldo de capital, sin intereses (*) | Eliminada por redundancia |
| `saldo_mora` | Numérica | Saldo en mora reportado | Base de `tiene_mora_reportada` |
| `saldo_mora_codeudor` | Numérica | Saldo en mora del codeudor | Eliminada (sin varianza) |
| `total_otros_prestamos` | Numérica | Suma de otras obligaciones | Base de `carga_deuda_previa` |
| `creditos_sectorFinanciero` | Numérica | Obligaciones en el sector financiero | Feature numérica |
| `creditos_sectorCooperativo` | Numérica | Obligaciones en el sector cooperativo | Feature numérica |
| `creditos_sectorReal` | Numérica | Obligaciones en el sector real | Feature numérica |

### Variable objetivo

| Variable | Tipo | Descripción | Distribución |
|---|---|---|---|
| `Pago_atiempo` | Binaria | 1 = pagada en su totalidad; 0 = incumplimiento | 95,25% / 4,75% |

---

## Variables derivadas

| Variable | Cálculo | Significado |
|---|---|---|
| `esfuerzo_pago` | `cuota_pactada / salario_cliente` | Capacidad de endeudamiento. Ratio cuota-ingreso del análisis de capacidad de pago del SARC |
| `saldo_relativo_ingreso` | `saldo_total / salario_cliente` | Peso de las obligaciones vigentes frente a la capacidad mensual |
| `carga_deuda_previa` | `total_otros_prestamos / salario_cliente` | Apalancamiento acumulado antes de este crédito |
| `intensidad_busqueda_credito` | `huella_consulta / (cant_creditosvigentes + 1)` | Consultas por crédito vigente. Distingue la actividad normal de un cliente bancarizado de la búsqueda urgente de liquidez |
| `tiene_mora_reportada` | `saldo_mora > 0` | Indicador binario de mora vigente |
| `rango_datacredito` | Discretización del score en cuatro tramos | Nivel de riesgo externo, ordinal |

Los ratios se calculan sobre valores en pesos, antes de la transformación
logarítmica. Cuando el denominador es nulo o cero, el ratio queda nulo y **no
se completa con cero**: un cliente cuya cuota supera su ingreso quedaría
marcado con el valor de menor riesgo posible, invirtiendo su perfil.

---

## Reglas de validación de las variables

Las reglas provienen del contexto normativo colombiano, **no de los rangos
observados en el dataset**. Un valor que incumple una regla se marca como nulo
y se imputa; el registro nunca se elimina.

| Variable | Regla | Fundamento |
|---|---|---|
| `edad_cliente` | [18, 100] | 18 años es la edad de capacidad legal para contratar (Ley 27 de 1977) |
| `salario_cliente` | [1, 100 SMMLV] | Un ingreso de 0 es un dato ausente. El techo equivale a $142.350.000 |
| `salario_cliente` (empleados) | ≥ 1 SMMLV | A un empleado formal no puede pagársele menos del mínimo legal |
| `puntaje_datacredito` | [150, 950] | Escala oficial de Datacrédito-Experian |
| `total_otros_prestamos` | [0, 500 SMMLV] | Saldo acumulado; admite un múltiplo mayor por no ser un flujo mensual |
| `capital_prestado` | > 0 | Un desembolso nulo o negativo no existe |
| `plazo_meses` | [1, 180] | Cubre incluso crédito por libranza a 15 años |
| `cuota_pactada` | > 0 | Una cuota nula no corresponde a un crédito |
| `saldo_mora` | ≥ 0 y ≤ `saldo_total` | La mora es una porción del saldo |
| `saldo_total`, `huella_consulta`, conteos | ≥ 0 | Saldos y conteos no admiten negativos |
| `tipo_laboral` | {Empleado, Independiente} | Dominio cerrado |
| `tendencia_ingresos` | {Creciente, Decreciente, Estable} | Dominio cerrado |

### Fuentes

- Superintendencia Financiera de Colombia. Circular Básica Contable y
  Financiera (Circular Externa 100 de 1995), Capítulo II — SARC.
- Ley 1527 de 2012, artículo 3 numeral 5 — tope del 50% para descuentos por
  libranza, ratificado por la Corte Constitucional en la sentencia T-864 de 2014.
- Decreto 1572 de 2024 — SMMLV 2025 ($1.423.500). Para 2026: $1.750.905.
- Ley 27 de 1977 — mayoría de edad a los 18 años.
- Datacrédito-Experian — escala del puntaje crediticio, 150 a 950 puntos.

Los tramos de `rango_datacredito` (150–499 alto riesgo, 500–649 regular,
650–799 bueno, 800–950 excelente) son una **convención de mercado**: la central
publica la escala pero no define tramos oficiales, y las fuentes consultadas
difieren entre sí. Lo relevante es que los cortes provienen de fuera del
dataset y no de sus cuartiles.

---

## Regla de coherencia entre calidad y negocio

Una condición puede ser simultáneamente indicio de error de captura y señal de
riesgo. El caso concreto: una cuota superior al ingreso del cliente.

En los datos históricos, los 35 casos correspondían a clientes con el ingreso
sin registrar (ceros y un valor de $1.000), de modo que tratarlos como error
era correcto. Pero al probar la interfaz se detectó que una solicitud nueva con
cuota de $15.000.000 e ingreso de $50.000 resultaba **aprobada**: la regla de
calidad anulaba la cuota, el ratio quedaba nulo y la regla de negocio nunca
llegaba a evaluarlo.

La corrección distingue ambos casos: la cuota se anula solo cuando el ingreso
mismo es inválido. El episodio ilustra por qué una regla de calidad y una regla
de negocio no pueden compartir la misma condición — una descarta el dato, la
otra necesita leerlo.

---

## Metodología de preparación de datos

Se sigue el proceso de 11 pasos del curso.

| # | Paso | Implementación |
|---|---|---|
| 1 | Integración de los datos | `CastTypes`, `DropDuplicates` |
| 2 | Eliminar irrelevantes y redundantes | `DropColumns` en dos momentos |
| 3 | Descripción estadística | `comprension_eda.ipynb` |
| 4 | Limpieza de atípicos | `FlagOutliersAsNull`, `ReglasCoherencia` |
| 5 | Limpieza de nulos | `SimpleImputer` dentro de cada rama |
| 6 | Creación de nuevas variables | `NuevasVariables`, `DiscretizarScore` |
| 7 | Correlaciones para redundancias | Umbral 0,70 |
| 8 | Correlaciones para irrelevancias | Umbral 0,10 |
| 9 | Reducción de variables (PCA) | **Omitido — ver justificación** |
| 10 | Balanceo de datos | SMOTE 1:2, solo sobre entrenamiento |
| 11 | Transformación según el método | Discretización, log, escalado, dummies, ordinal |

### Por qué se omitió el PCA

Los componentes principales son combinaciones lineales de las variables
originales y no tienen interpretación de negocio. El modelo heurístico opera
sobre variables en unidades reales ("esfuerzo de pago superior al 50%"), y esas
reglas no pueden escribirse sobre componentes abstractos. La reducción de
dimensionalidad se logró por eliminación justificada de variables.

### Escalado: StandardScaler en lugar de Min-Max

La presentación del curso ilustra la normalización con Max-Min a rango [0,1].
Se optó por `StandardScaler` porque las variables monetarias presentan
asimetrías fuertes (skewness superior a 38): Min-Max comprimiría el 95% de los
registros contra el cero. La transformación logarítmica previa corrige la forma
de la distribución; el escalado corrige el rango. Son problemas distintos.

---

## Prevención de fuga de información

| Riesgo | Tratamiento |
|---|---|
| `puntaje` | Eliminada. 87% de los registros en un único valor, IQR de 0, correlación muy alta con el objetivo y mínimo de −38: el comportamiento corresponde a un score predictivo ya calculado |
| Imputación y escalado | Se ajustan dentro del `ColumnTransformer`, que vive en el pipeline del modelo y se reajusta en cada pliegue de la validación cruzada |
| Balanceo | SMOTE se aplica dentro de cada pliegue, no antes de la partición. De lo contrario, registros sintéticos derivados de filas de validación terminarían en entrenamiento |
| Evaluación | El conjunto de prueba se usa una sola vez, al final, y conserva la prevalencia original (4,71%) |

`puntaje_datacredito` sí se conserva: es un score de central consultado antes de
otorgar el crédito, disponible legítimamente al decidir.

Se verificó explícitamente que `saldo_total` no constituye fuga: la tasa de
incumplimiento es 6% entre los registros con saldo cero y 5% entre los que
tienen saldo positivo. Si el saldo reflejara el estado final del crédito, la
diferencia sería drástica.

---

## Hallazgos del análisis

**Sin `puntaje`, la señal predictiva del dataset es baja.** Ninguna variable
supera el umbral de relevancia de 0,10 salvo `rango_datacredito` (0,1379). El
umbral del curso está pensado para un objetivo continuo; con un objetivo
binario desbalanceado 95/5 el coeficiente de Spearman está estructuralmente
comprimido, por lo que el criterio se aplica como diagnóstico y no como regla
de eliminación automática.

**Las señales de comportamiento predicen mejor que las de perfil.** El orden de
asociación con el objetivo lo encabezan `rango_datacredito` (0,1379),
`tiene_mora_reportada` (0,0992) e `intensidad_busqueda_credito` (0,0754). Al
final aparecen `salario_cliente` (0,0008) y `tipo_credito` (0,0009). Lo que
anticipa el incumplimiento es cómo se ha comportado la persona con el crédito,
no cuánto gana.

**Los ingresos declarados no son confiables.** Percentil 99 de $39,6 millones
frente a percentil 99,9 de $3.344 millones: un salto de 85 veces, incompatible
con una distribución natural. El máximo observado ($22.000 millones mensuales)
equivale a más de 15.000 salarios mínimos. Esto explica en parte la baja
asociación de `salario_cliente` y arrastra a las variables derivadas que lo
usan como denominador.

**150 registros con edad entre 121 y 123 años.** La concentración en tres años
consecutivos —y no dispersa— es compatible con una fecha de nacimiento por
defecto en el sistema origen, no con errores de digitación.

**`saldo_mora_codeudor` no tiene varianza utilizable:** 3 valores positivos en
10.170 observaciones.

---

## Variables eliminadas

| Variable | Motivo |
|---|---|
| `puntaje` | Fuga de información |
| `fecha_prestamo` | El objetivo no es un análisis de series de tiempo |
| `tendencia_ingresos` | 27,78% de nulos |
| `promedio_ingresos_datacredito` | 27,22% de nulos. Se utilizó previamente para recuperar 15 salarios registrados en cero |
| `saldo_principal` | Contenida en `saldo_total`, con más nulos |
| `saldo_mora_codeudor` | 3 valores distintos de cero |
| `cuota_pactada` | Consumida por `esfuerzo_pago` |
| `cant_creditosvigentes` | Redundante con `creditos_sectorFinanciero`; menor asociación con el objetivo |
| `puntaje_datacredito` | Reemplazada por `rango_datacredito` |
| `saldo_total` | Redundancia de 0,94 con `saldo_relativo_ingreso` |
| `total_otros_prestamos` | Redundancia de 0,75 con `carga_deuda_previa` |

Sobre el umbral de nulos del 20%: por encima de esa proporción la imputación
deja de completar información y pasa a fabricar la distribución de la variable.
Además, los nulos de las variables de ingreso no son aleatorios —un cliente sin
ingresos reportados en central no es un cliente promedio—, de modo que
imputarlos introduciría sesgo.

Se conservan únicamente los tipos de crédito 4 y 9, que concentran el 98,7% de
la muestra: 140 registros excluidos.

---

## Resultado del feature engineering

- 10.623 registros tras el filtro de alcance
- 16 variables predictoras + objetivo
- 18 features tras la codificación (12 numéricas, 4 dummies, 1 ordinal, 1 binaria)
- Partición estratificada 70/30: 7.436 entrenamiento, 3.187 evaluación
- Tasa de incumplimiento en evaluación: 4,71%, igual a la del dataset original

`ft_engineering.py` expone `generar_datasets()`, que retorna los conjuntos en
memoria junto con el preprocesador y el DataFrame limpio sin codificar. No se
escriben archivos intermedios: evita que el entrenamiento consuma un resultado
desactualizado del pipeline.

---

## Modelo heurístico

Cuatro reglas duras combinadas con lógica **OR**: basta que una se active para
rechazar. Esto distingue un sistema de reglas duras de un árbol de decisión,
donde las condiciones se encadenan con AND: una mora vigente no deja de ser
motivo de rechazo porque el cliente cumpla el resto de criterios.

| Regla | Condición | Fundamento |
|---|---|---|
| `mora_vigente` | `tiene_mora_reportada = 1` | Condición de rechazo más estándar del sector. El SARC exige evaluar el comportamiento de pago vigente |
| `score_alto_riesgo` | `rango_datacredito = Alto_riesgo` | Tramo 150–499 de la escala de Datacrédito |
| `esfuerzo_pago_excesivo` | `esfuerzo_pago > 0,50` | Ley 1527 de 2012, art. 3 num. 5 |
| `edad_fuera_de_politica` | `edad_cliente > 85` | Edad máxima usual para deudores de consumo |

**Salvedad sobre la regla de esfuerzo de pago:** el tope legal del 50% aplica
sobre salario neto y en el contexto de libranza. El ingreso del dataset es
probablemente bruto, lo que hace la regla más permisiva que la norma, no más
estricta. Para una regla dura esa dirección es la correcta.

**La regla de edad no se activa en esta muestra** (cero registros entre 85 y
100 años). Se conserva porque en producción sí aplicaría.

### Aporte de cada regla

| Regla | Marcados | Incumplieron | Precisión | Lift |
|---|---|---|---|---|
| `score_alto_riesgo` | 13 | 9 | 0,6923 | 14,75 |
| `mora_vigente` | 38 | 15 | 0,3947 | 8,41 |
| `esfuerzo_pago_excesivo` | 16 | 0 | 0,0000 | 0,00 |
| `edad_fuera_de_politica` | 0 | — | — | — |

**Solapamiento:** los 13 clientes marcados por `score_alto_riesgo` son los
mismos que ya tienen mora vigente. Su aporte incremental es cero en esta
muestra, aunque en producción puede detectar historiales deteriorados sin mora
actual.

**Sobre `esfuerzo_pago_excesivo`:** cero incumplimientos en 16 rechazos. Con una
tasa base del 4,7%, la probabilidad de observar cero incumplimientos en 16
clientes al azar es del 46%, de modo que no hay evidencia suficiente para
descartar la regla. Se conserva por su fundamento legal.

**Experimento de calibración:** se evaluó bajar el umbral de 0,50 (tope legal
de libranza) a 0,30 (práctica comercial habitual). El criterio de aceptación se
fijó antes de ver el resultado: mantener el lift por encima de 1. La regla pasó
de lift 0,00 a 0,30 —ambos por debajo de 1— y los rechazos erróneos se
triplicaron. Se revirtió el cambio.

---

## Comparación de modelos

Siete modelos evaluados con validación cruzada estratificada repetida
(5 pliegues × 3 repeticiones = 15 mediciones por modelo).

| Modelo | PR-AUC | Desv. | ROC-AUC | Ajuste (s) | CV |
|---|---|---|---|---|---|
| Regresión logística | **0,1328** | 0,0380 | 0,6409 | 0,14 | 0,2861 |
| HistGradientBoosting | 0,1004 | 0,0183 | 0,6009 | 1,48 | 0,1823 |
| Random Forest | 0,0997 | 0,0227 | 0,6513 | 0,90 | 0,2277 |
| Red neuronal | 0,0985 | 0,0310 | 0,5810 | 4,02 | 0,3147 |
| AdaBoost | 0,0924 | 0,0199 | 0,6299 | 2,07 | 0,2154 |
| KNN | 0,0767 | 0,0127 | 0,6125 | 0,10 | 0,1656 |
| Árbol de decisión | 0,0640 | 0,0088 | 0,5819 | 0,16 | 0,1375 |

Las tres dimensiones del enunciado: **desempeño** (PR-AUC medio),
**consistencia** (desviación y coeficiente de variación entre mediciones),
**escalabilidad** (tiempo de ajuste y predicción).

### Por qué PR-AUC y no exactitud ni F1

Con 4,7% de positivos, la exactitud premia aprobar a todos: un modelo trivial
alcanza 95,3%. F1 depende de un umbral de corte que en este punto todavía no se
ha decidido, y con este desbalance casi ningún cliente supera 0,5 de
probabilidad. PR-AUC evalúa la capacidad de ordenar clientes por riesgo con
independencia del corte, que es lo que debe decidir la selección del modelo.

Frente a ROC-AUC: la curva ROC usa la tasa de falsos positivos, calculada sobre
3.037 buenos pagadores, de modo que cientos de falsos positivos apenas la
mueven. La curva PR usa la precisión, que sí se desploma. Con clases raras,
PR-AUC es más honesta.

### Selección estadística

ANOVA sobre las 15 mediciones: F = 12,55, p < 0,000001. Tukey HSD confirma que
la regresión logística difiere significativamente de todos los demás modelos.

**Limitación documentada:** el ANOVA supone observaciones independientes, y los
pliegues de una validación cruzada comparten datos entre sí. El p-valor
resultante es optimista. El contraste se aplica porque el enunciado lo exige y
porque sigue siendo útil para ordenar, pero no constituye una prueba formal de
superioridad.

### Hiperparametrización

Finalistas: regresión logística, HistGradientBoosting, Random Forest. Cada uno
optimizado con GridSearch y con optimización bayesiana (Optuna, 40 intentos),
ambos maximizando PR-AUC con validación cruzada interna sobre entrenamiento.

**La optimización aportó poco:** la regresión logística pasó de 0,1328 a 0,1316.
El límite no está en los hiperparámetros sino en la señal disponible. Prueba
adicional: entre corridas, el valor óptimo de `C` varió de 0,29 a 0,98 con
diferencias de PR-AUC menores al 1,5%, lo que indica una superficie de
optimización prácticamente plana.

### Regla de desempate

En una corrida, tras la optimización la red neuronal superó a la regresión
logística por 0,0019 de PR-AUC, una diferencia catorce veces menor que la
desviación estándar de la métrica. Como el máximo de una búsqueda de 40
intentos está sesgado al alza —y más aún para modelos con espacios de búsqueda
amplios—, se implementó una regla explícita: cuando dos configuraciones quedan
dentro de 0,01, se desempata por el desempeño en validación cruzada, que no
participó de esa búsqueda. El conjunto de prueba confirmó la decisión.

---

## Modelo seleccionado

**Regresión logística**, optimizada con búsqueda bayesiana (C = 0,9847).

### Desempeño sobre el conjunto de evaluación (3.187 registros)

| Métrica | Heurístico | Regresión logística |
|---|---|---|
| PR-AUC | — | 0,1074 |
| ROC-AUC | 0,5062 | 0,6484 |
| Precisión (incumplimiento) | 0,1154 | 0,1127 |
| Recall (incumplimiento) | 0,0200 | 0,1600 |
| F1 (incumplimiento) | 0,0341 | 0,1322 |
| Exactitud | 0,9467 | 0,9012 |

El heurístico no produce probabilidades, por lo que no tiene PR-AUC.

### Interpretación

Con una tasa base de incumplimiento del 4,71%, un PR-AUC de 0,1074 significa
que el modelo distingue riesgo **2,3 veces mejor que el azar**. Es señal real,
pero por debajo de lo que se considera utilizable para automatizar decisiones
de crédito: un scorecard con ROC-AUC de 0,65 está en el rango bajo de la
industria, donde el mínimo aceptable ronda 0,70.

El límite está en los datos, no en el modelado. Tres evidencias: los siete
modelos rindieron parecido, la optimización aportó menos del 1%, y ninguna
variable supera 0,14 de asociación con el objetivo tras eliminar la fuga.

**El modelo es apto como insumo de priorización para revisión manual, no como
decisor automático.** Mejorar el desempeño requiere mejores datos —en
particular, ingresos verificados— antes que mejores algoritmos.

---

## Balanceo: efecto sobre la calibración

Se compararon dos configuraciones de SMOTE sobre el conjunto de entrenamiento.

| | SMOTE 1:1 | SMOTE 1:2 |
|---|---|---|
| Registros sintéticos | ~6.738 | ~3.543 |
| PR-AUC en test | 0,1095 | 0,1074 |
| Umbral óptimo de costo | 0,75 | 0,60 |
| Precisión con umbral 0,50 | 0,0677 | 0,1127 |
| Buenos clientes rechazados | 1.129 | 189 |

El PR-AUC se mantuvo prácticamente igual, confirmando que el ordenamiento de
riesgo no depende del balanceo. Lo que cambió fue la **calibración**: generar
6.738 registros sintéticos a partir de 349 casos reales implica replicar cada
moroso unas 19 veces, lo que desplaza las probabilidades al alza y obliga a
cortar en el extremo superior de la escala.

Se adoptó el balanceo parcial (1:2).

---

## Umbral de corte

El umbral por defecto de 0,5 supone que ambos errores cuestan lo mismo. No es
el caso: rechazar a un cliente viable cuesta el margen del crédito, mientras
que aprobar a un moroso cuesta la porción no recuperada del capital.

**Supuestos de costo** (declarados como tales; no provienen de datos de una
entidad): margen de intermediación neto del 10% sobre el capital; severidad de
la pérdida del 65%. Relación resultante: un impago cuesta como 6,5 clientes
buenos rechazados. El costo se calcula con el capital real de cada registro.

| Umbral | Rechazo | Morosos detectados | Precisión | Costo total |
|---|---|---|---|---|
| 0,45 | 12,6% | 97 de 349 | 0,1036 | $644,6 M |
| **0,50** | **6,4%** | **70 de 349** | **0,1477** | **$570,4 M** |
| 0,55 | 3,3% | 53 de 349 | 0,2190 | $535,3 M |
| 0,60 | 1,6% | 36 de 349 | 0,3025 | $533,6 M |

**Análisis de sensibilidad:** con ratio de costos de 3,3 el óptimo está en 0,65;
con 6,5 en 0,60; con 13,3 en 0,40. El resultado depende de los supuestos, que
por eso se declaran.

**Decisión: umbral 0,50.** El mínimo estricto de costo está en 0,60, pero
detecta solo 36 de 349 morosos (10%), cifra difícil de sostener ante un
supervisor. Con 0,50 se detecta el doble, rechazando un 6,4% de la cartera
manejable para revisión manual, con un costo 6,9% superior al mínimo.

El umbral se seleccionó sobre el conjunto de entrenamiento. El conjunto de
prueba se reservó para la evaluación final.

---

## Despliegue

`model_deploy.py` implementa la decisión combinada:

```
solicitud
    ↓
limpieza y variables derivadas
    ↓
reglas duras ── activa ──→ RECHAZADO_POR_REGLA (motivo auditable)
    ↓
ninguna activa
    ↓
modelo → probabilidad → umbral → APROBADO / RECHAZADO_POR_MODELO
```

Las solicitudes fuera del alcance declarado del modelo (tipos de crédito
distintos de 4 y 9) se reportan como `FUERA_DE_ALCANCE` en lugar de
descartarse en silencio: en producción una solicitud no puede desaparecer sin
respuesta.

**El pipeline de limpieza se ajusta una sola vez y se guarda dentro del
artefacto.** Reajustarlo con cada lote haría que decisiones aprendidas —como
qué columnas superan el umbral de nulos— dependieran del lote recibido: un
envío con datos incompletos podría eliminar una variable que el modelo espera,
sin que nada falle de forma visible.

`app.py` expone la interfaz en Streamlit, con dos modos: carga de archivo para
evaluación por lotes y formulario para solicitudes individuales. Ambos aplican
exactamente la misma lógica.

---

## Monitoreo y datadrift

En riesgo crediticio el resultado real se conoce meses después del desembolso,
cuando la obligación vence o se paga. No es posible medir el acierto del modelo
de forma continua, así que el monitoreo vigila lo que sí es observable: que la
población de entrada siga pareciéndose a la de entrenamiento, y que la
distribución de decisiones se mantenga estable.

**Medidas:** PSI (Population Stability Index) para variables numéricas, V de
Cramér para categóricas, y prueba de Kolmogorov-Smirnov como complemento.
Umbrales de interpretación estándar en la industria: por debajo de 0,10
estable; entre 0,10 y 0,25 deriva moderada; por encima de 0,25 deriva
significativa.

El PSI decide y el valor p acompaña: con muestras grandes, la prueba de
Kolmogorov-Smirnov declara significativa cualquier diferencia, por irrelevante
que sea.

### Verificación del sistema de alertas

Sin datos futuros no hay forma de comprobar que el monitoreo detecta un cambio
poblacional. Se implementó una simulación que desplaza deliberadamente cuatro
variables, reproduciendo un escenario de relajación de los criterios de
originación.

| | Población comparable | Deriva inducida |
|---|---|---|
| Variables críticas | 0 | 3 |
| Variables en alerta | 0 | 2 |
| PSI máximo | 0,0378 | 0,5814 |
| Tasa de rechazo | 6,4% | 40,5% |
| Conclusión | Población estable | Reentrenamiento recomendado |

Que el lote normal no produzca falsas alarmas es tan importante como que el
derivado las dispare: un monitoreo hipersensible se ignora a las pocas semanas.

**La tasa de rechazo resultó el indicador más sensible**, con un salto mucho más
visible que cualquier PSI individual. Confirma que el efecto conjunto de las
variables se detecta antes que las variables por separado.

---

## Distinción entre tipos de regla

- **Reglas de calidad** — determinan si un dato es válido. Se aplican en el
  feature engineering. Un valor fuera de regla se convierte en nulo.
- **Reglas de negocio** — determinan condiciones de decisión sobre datos ya
  válidos.
- **Reglas duras** — condiciones que prevalecen sobre la predicción del modelo.
  Se justifican por política de riesgo, no por su poder separador en la muestra.

La edad máxima de 85 años ilustra la diferencia: como regla de calidad
convertiría en nulo el dato de un cliente de 88 años y lo imputaría con la
mediana, inventando una edad falsa. Como regla de negocio, el dato se conserva
y el cliente se rechaza con una razón auditable.

---

## Automatización de pruebas

El análisis de código se ejecuta con SonarCloud mediante GitHub Actions en cada
push y pull request. Los indicadores de calidad, seguridad y estilo se evalúan
sobre `mlops_pipeline/src`; los notebooks se excluyen por contener salidas y
gráficos embebidos que no son código fuente evaluable.

El Quality Gate reporta fallo por cobertura de pruebas del 0%: el alcance del
entregable era generar el reporte de análisis, no desarrollar la suite de
pruebas unitarias. Los demás indicadores sí reflejan el estado del código.

---

## Limitaciones reconocidas

- **Calidad de los ingresos declarados.** Los valores extremos y los registros
  en cero comprometen `salario_cliente` y las cuatro variables derivadas que lo
  usan como denominador. Es la hipótesis más plausible sobre la baja señal del
  dataset.
- **Interpretación de `saldo_total` y `saldo_principal`.** El diccionario se
  construyó por inferencia; no hay documentación del proveedor que la confirme.
- **Independencia en el ANOVA.** Los pliegues de la validación cruzada comparten
  datos, de modo que el p-valor es optimista.
- **Supuestos de costo.** El margen y la severidad no provienen de datos de una
  entidad real. El análisis de sensibilidad muestra que el umbral óptimo varía
  entre 0,40 y 0,65 según el escenario.
- **Cobertura de pruebas.** Sin pruebas unitarias, los errores silenciosos se
  detectan solo por inspección manual de los resultados.

---

## Estado del proyecto

- [x] Carga de datos
- [x] Análisis exploratorio
- [x] Feature engineering
- [x] Modelo heurístico
- [x] Entrenamiento y comparación de modelos
- [x] Evaluación y selección del umbral
- [x] Despliegue
- [x] Monitoreo y datadrift
- [x] Análisis de código con SonarCloud
