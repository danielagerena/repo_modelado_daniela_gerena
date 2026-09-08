# Modelo de riesgo crediticio — cartera de consumo (Colombia)

Pipeline de MLOps para predecir el cumplimiento de pago de obligaciones de
crédito de consumo, a partir de información disponible en el momento de la
decisión crediticia.

**Objetivo:** `Pago_atiempo` — 1 si la obligación fue pagada en su totalidad,
0 si no. La variable se registra una vez cerrada la obligación, ya sea por
vencimiento del plazo o por prepago del cliente.

**Autora:** Daniela Gerena
**Curso:** Ciencia de Datos en Producción — Universidad Pontificia Bolivariana

---

## Estructura del repositorio

```
repo_modelado_daniela_gerena/
├── mlops_pipeline/
│   ├── src/
│   │   ├── Cargar_datos.ipynb      Materialización de la fuente
│   │   ├── comprension_eda.ipynb   Análisis exploratorio
│   │   ├── ft_engineering.py       Limpieza, features y partición
│   │   ├── hueristic_model.py      Reglas de negocio y reglas duras
│   │   ├── model_training.py       Entrenamiento y comparación
│   │   ├── model_deploy.py         Despliegue en endpoint
│   │   ├── model_evaluation.py     Métricas del modelo desplegado
│   │   └── model_monitoring.py     Monitoreo y datadrift
│   └── config.json                 Parametrización del pipeline
├── Base_de_datos.csv               Dataset (10.763 registros)
├── requirements.txt
├── set_up.bat                      Configuración del entorno
└── readme.md
```

---

## Instalación

```bat
set_up.bat
```

Crea el entorno virtual e instala las dependencias. Luego:

```powershell
.venv\Scripts\activate
python mlops_pipeline\src\ft_engineering.py
```

Toda la parametrización vive en `mlops_pipeline/config.json`. El código no
contiene valores fijos: rutas, umbrales, reglas de calidad y listas de
variables se editan ahí sin tocar los scripts.

---

## Diccionario de datos

El dataset original tiene 10.763 registros y 23 variables. El diccionario se
construyó a partir del análisis exploratorio y del funcionamiento del sistema
de crédito colombiano; no se entregó documentación de origen. Las
interpretaciones marcadas con (*) son inferencias validadas contra el
comportamiento de los datos.

### Variables del crédito

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `tipo_credito` | Categórica | Línea o producto de crédito. Códigos 4 (72%) y 9 (27%) concentran el 98,7% | Feature nominal |
| `fecha_prestamo` | Fecha | Fecha de desembolso. Rango 26/11/2024 – 26/04/2026 | Eliminada |
| `capital_prestado` | Numérica | Monto desembolsado, en pesos | Feature numérica |
| `plazo_meses` | Numérica | Plazo pactado en meses | Feature numérica |
| `cuota_pactada` | Numérica | Valor de la cuota periódica, en pesos | Base de `esfuerzo_pago` |

### Variables del cliente

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `edad_cliente` | Numérica | Edad en años al momento del desembolso | Feature numérica |
| `tipo_laboral` | Categórica | Vinculación: Empleado o Independiente | Feature nominal |
| `salario_cliente` | Numérica | Ingreso mensual declarado, en pesos | Feature numérica |
| `promedio_ingresos_datacredito` | Numérica | Ingreso promedio reportado por la central (*) | Eliminada por nulos |
| `tendencia_ingresos` | Categórica | Evolución del ingreso reportado: Creciente, Decreciente, Estable | Eliminada por nulos |

### Variables de comportamiento crediticio (central de riesgo)

| Variable | Tipo | Descripción | Uso final |
|---|---|---|---|
| `puntaje` | Numérica | Score interno de la entidad | **Eliminada (fuga de datos)** |
| `puntaje_datacredito` | Numérica | Score de central de riesgo, escala 150–950 | Base de `rango_datacredito` |
| `cant_creditosvigentes` | Numérica | Total de obligaciones vigentes del cliente | Base de `intensidad_busqueda_credito` |
| `huella_consulta` | Numérica | Consultas recientes del historial en centrales | Feature numérica |
| `saldo_total` | Numérica | Saldo pendiente reportado, incluidos intereses y cargos (*) | Base de `saldo_relativo_ingreso` |
| `saldo_principal` | Numérica | Saldo pendiente de capital, sin intereses (*) | Eliminada por redundancia |
| `saldo_mora` | Numérica | Saldo en mora reportado, en pesos | Base de `tiene_mora_reportada` |
| `saldo_mora_codeudor` | Numérica | Saldo en mora del codeudor | Eliminada (sin varianza, comportamiento casi constante) |
| `total_otros_prestamos` | Numérica | Suma de otras obligaciones del cliente | Base de `carga_deuda_previa` |
| `creditos_sectorFinanciero` | Numérica | Obligaciones en el sector financiero | Feature numérica |
| `creditos_sectorCooperativo` | Numérica | Obligaciones en el sector cooperativo | Feature numérica |
| `creditos_sectorReal` | Numérica | Obligaciones en el sector real | Feature numérica |

### Variable objetivo

| Variable | Tipo | Descripción | Distribución |
|---|---|---|---|
| `Pago_atiempo` | Binaria | 1 = obligación pagada en su totalidad; 0 = incumplimiento | 95,25% / 4,75% |

---

## Variables derivadas

| Variable | Cálculo | Significado |
|---|---|---|
| `esfuerzo_pago` | `cuota_pactada / salario_cliente` | Capacidad de endeudamiento. Ratio cuota-ingreso del análisis de capacidad de pago exigido por el SARC |
| `saldo_relativo_ingreso` | `saldo_total / salario_cliente` | Peso de las obligaciones vigentes frente a la capacidad mensual |
| `carga_deuda_previa` | `total_otros_prestamos / salario_cliente` | Apalancamiento acumulado antes de este crédito |
| `intensidad_busqueda_credito` | `huella_consulta / (cant_creditosvigentes + 1)` | Consultas por crédito vigente. Distingue la actividad normal de un cliente bancarizado de la búsqueda urgente de liquidez |
| `tiene_mora_reportada` | `saldo_mora > 0` | Indicador binario de mora vigente en central |
| `rango_datacredito` | Discretización del score en cuatro tramos | Nivel de riesgo externo en categorías ordinales |

Los ratios se calculan sobre valores en pesos, antes de cualquier
transformación logarítmica. Ninguna utiliza información posterior a la
decisión crediticia.

---

## Reglas de validación de las variables

Las reglas provienen del contexto normativo y del funcionamiento del sistema
crediticio colombiano, **no de los rangos observados en el dataset**. Un valor
que incumple una regla se marca como nulo y se imputa; el registro nunca se
elimina.

| Variable | Regla | Fundamento |
|---|---|---|
| `edad_cliente` | [18, 100] | 18 años es la edad de capacidad legal para contratar (Ley 27 de 1977). El límite superior excede la esperanza de vida en Colombia |
| `salario_cliente` | [1, 100 SMMLV] | Un ingreso de 0 es un dato ausente, no un salario. El techo equivale a $142.350.000 |
| `salario_cliente` (empleados) | ≥ 1 SMMLV | A un empleado formal no puede pagársele menos del mínimo legal ($1.423.500 en 2025, Decreto 1572 de 2024) |
| `puntaje_datacredito` | [150, 950] | Escala oficial de Datacrédito-Experian |
| `total_otros_prestamos` | [0, 500 SMMLV] | Saldo acumulado; admite un múltiplo mayor por no ser un flujo mensual |
| `capital_prestado` | > 0 | Un desembolso nulo o negativo no existe |
| `plazo_meses` | [1, 180] | Cubre incluso crédito por libranza a 15 años |
| `cuota_pactada` | > 0 y ≤ `salario_cliente` | Una cuota superior al ingreso total es un error de registro |
| `saldo_mora` | ≥ 0 y ≤ `saldo_total` | La mora es una porción del saldo y no puede excederlo |
| `saldo_total`, `huella_consulta`, `cant_creditosvigentes`, `creditos_sector*` | ≥ 0 | Saldos y conteos no admiten valores negativos |
| `tipo_laboral` | {Empleado, Independiente} | Dominio cerrado |
| `tendencia_ingresos` | {Creciente, Decreciente, Estable} | Dominio cerrado |

### Fuentes

- Superintendencia Financiera de Colombia. Circular Básica Contable y
  Financiera (Circular Externa 100 de 1995), Capítulo II — Sistema de
  Administración del Riesgo Crediticio (SARC).
- Decreto 1572 de 2024 — salario mínimo mensual legal vigente 2025
  ($1.423.500). Para 2026 el valor es $1.750.905.
- Ley 27 de 1977 — mayoría de edad a los 18 años.
- Datacrédito-Experian — escala del puntaje crediticio, 150 a 950 puntos.

Los tramos de `rango_datacredito` (150–499 alto riesgo, 500–649 regular,
650–799 bueno, 800–950 excelente) son una **convención de mercado**: la
central publica la escala pero no define tramos oficiales, y las fuentes
consultadas difieren entre sí. Lo relevante metodológicamente es que los
cortes provienen de fuera del dataset y no de sus cuartiles.

---

## Metodología

Se sigue el proceso de preparación de datos del curso (11 pasos).

| # | Paso | Implementación |
|---|---|---|
| 1 | Integración de los datos | `CastTypes`, `DropDuplicates` |
| 2 | Eliminar irrelevantes y redundantes | `DropColumns` en dos momentos |
| 3 | Descripción estadística | `comprension_eda.ipynb` |
| 4 | Limpieza de datos atípicos | `FlagOutliersAsNull`, `ReglasCoherencia` |
| 5 | Limpieza de datos nulos | `SimpleImputer` dentro de cada rama |
| 6 | Creación de nuevas variables | `NuevasVariables`, `DiscretizarScore` |
| 7 | Correlaciones para redundancias | Umbral 0,70 |
| 8 | Correlaciones para irrelevancias | Umbral (flexibilizado) 0,10 |
| 9 | Reducción de variables (PCA) | **Omitido — ver justificación** |
| 10 | Balanceo de datos | SMOTE, solo sobre set de entrenamiento |
| 11 | Transformación según el método | Discretización, escalado, dummies, ordinal |

### Por qué se omitió el PCA

Los componentes principales son combinaciones lineales de las variables
originales y no tienen interpretación de negocio. El proyecto contempla un
modelo heurístico de reglas explícitas que opera sobre variables en unidades
reales ("esfuerzo de pago superior a X"), y esas reglas no pueden escribirse
sobre componentes abstractos. La reducción de dimensionalidad se logró por
eliminación justificada de variables, no por proyección.

### Escalado: StandardScaler en lugar de Min-Max

La presentación del curso ilustra la normalización con Max-Min a rango [0,1].
Se optó por `StandardScaler` porque las variables monetarias del dataset
presentan asimetrías fuertes (skewness superior a 38): Min-Max comprimiría el
95% de los registros contra el cero. La transformación logarítmica previa
sobre las monetarias corrige la forma de la distribución; el escalado corrige
el rango. Son problemas distintos.

---

## Prevención de fuga de información

| Riesgo | Tratamiento |
|---|---|
| `puntaje` | Eliminada. 87% de los registros en un único valor, IQR de 0, correlación muy alta con el objetivo y mínimo de −38: el comportamiento corresponde a un score predictivo ya calculado, no a un atributo observable |
| Imputación | Los estadísticos se aprenden dentro del `ColumnTransformer`, ajustado únicamente sobre entrenamiento |
| Balanceo | SMOTE se aplica después de la partición y solo sobre entrenamiento. Aplicarlo antes haría que registros sintéticos derivados de filas de evaluación terminaran en entrenamiento |
| Evaluación | El conjunto de prueba conserva la prevalencia original (4,71% de incumplimiento), que es la que verá el modelo en producción |

`puntaje_datacredito` sí se conserva: es un score de central consultado antes
de otorgar el crédito, disponible legítimamente en el momento de la decisión.

Se verificó explícitamente que `saldo_total` no constituye fuga: la tasa de
incumplimiento es 6% entre los registros con saldo cero y 5% entre los que
tienen saldo positivo. Si el saldo reflejara el estado final de este crédito,
la diferencia sería drástica.

---

## Hallazgos relevantes

**Sin `puntaje`, la señal predictiva del dataset es baja.** Ninguna variable
supera el umbral de relevancia de 0,10 salvo `rango_datacredito` (0,1379). El
umbral de la metodología está pensado para un objetivo continuo; con un
objetivo binario desbalanceado 95/5 el coeficiente de Spearman está
estructuralmente comprimido, por lo que el criterio se aplica como diagnóstico
y no como regla de eliminación automática.

**Las señales de comportamiento predicen mejor que las de perfil.** El orden
de asociación con el objetivo lo encabezan `rango_datacredito` (0,1379),
`tiene_mora_reportada` (0,0992) e `intensidad_busqueda_credito` (0,0754). Al
final aparecen `salario_cliente` (0,0008) y `tipo_credito` (0,0009). Lo que
anticipa el incumplimiento es cómo se ha comportado la persona con el crédito,
no cuánto gana.

**`tiene_mora_reportada` concentra señal pese a su escasez.** Solo 55 registros
tienen mora positiva (0,5%), y aun así es la segunda variable más asociada al
objetivo. Se conserva por su valor como regla dura, no como feature.

**150 registros con edad entre 121 y 123 años.** La concentración en tres años
consecutivos —y no dispersa por todo el rango— es compatible con una fecha de
nacimiento por defecto en el sistema origen, no con errores de digitación.

**Errores de escala en variables monetarias.** En `salario_cliente`, el
percentil 99 es $39,6 millones y el 99,9 es $3.344 millones: un salto de 85
veces. Una distribución natural crece de forma continua. El máximo observado
($22.000 millones mensuales) equivale a más de 15.000 salarios mínimos.

**`saldo_mora_codeudor` no tiene varianza utilizable:** 3 valores positivos en
10.170 observaciones. Ninguna transformación rescata señal de tres casos.

---

## Variables eliminadas y su justificación

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
deja de completar información y pasa a fabricar la distribución de la
variable. Además, los nulos de las variables de ingreso no son aleatorios —un
cliente sin ingresos reportados en central no es un cliente promedio—, de modo
que imputarlos introduciría sesgo.

Se conservan únicamente los tipos de crédito 4 y 9, que concentran el 98,7% de
la muestra. Las demás categorías quedan fuera del alcance declarado del modelo
en lugar de ser imputadas: 140 registros excluidos.

---

## Resultado del feature engineering

- 10.623 registros tras el filtro de alcance
- 16 variables predictoras + objetivo
- 18 features tras la codificación (12 numéricas, 4 dummies, 1 ordinal, 1 binaria)
- Partición estratificada 70/30: 7.436 registros de entrenamiento, 3.187 de evaluación
- Entrenamiento balanceado con SMOTE: 14.174 registros
- Tasa de incumplimiento en evaluación: 4,71%, igual a la del dataset original

`ft_engineering.py` expone `generar_datasets()`, que retorna los conjuntos en
memoria junto con el preprocesador ajustado y el DataFrame limpio sin
codificar. Este último es el insumo del modelo heurístico, que debe operar
sobre variables en unidades reales.

No se escriben archivos intermedios: evita que el entrenamiento consuma un
resultado desactualizado del pipeline.

---

## Nota sobre la métrica de evaluación

Con un desbalance de 95/5, la exactitud es engañosa: un modelo que prediga
"todos pagan" acierta el 95,3% sin detectar un solo incumplimiento. La
comparación de modelos se hará sobre recall, F1 y AUC-ROC de la clase
minoritaria.

---

## Arquitectura prevista

```
Datos → Feature Engineering → Reglas heurísticas → Modelo ML → Decisión combinada
```

El modelo heurístico opera sobre datos sin balancear y en unidades originales.
Un umbral calibrado sobre datos donde la clase minoritaria fue elevada
artificialmente al 50% sobreestima su precisión y subestima la tasa de falsos
rechazos.

### Candidatas a reglas duras

| Variable | Fundamento |
|---|---|
| `tiene_mora_reportada` | Una mora vigente en central es la condición de rechazo más estándar del sector |
| `esfuerzo_pago` | Capacidad de pago, criterio normativo del SARC |
| `rango_datacredito` | Un score en el tramo de alto riesgo refleja historial de incumplimiento |
| `carga_deuda_previa` | Nivel de endeudamiento, criterio explícito de evaluación en el SARC |
| `edad_cliente` | Edad máxima de política (85 años), parametrizada en `config.json` |
| `intensidad_busqueda_credito` | Búsqueda intensiva de crédito como señal de estrés de liquidez |

Los umbrales se definirán a partir de política de riesgo y fuentes externas,
no maximizando la separación en este dataset: un corte ajustado a la muestra
no generaliza.

---

## Distinción entre tipos de regla

- **Reglas de calidad** — determinan si un dato es válido. Se aplican en el
  feature engineering. Un valor fuera de regla se convierte en nulo.
- **Reglas de negocio** — determinan condiciones de decisión sobre datos ya
  válidos.
- **Reglas duras** — condiciones que prevalecen sobre la predicción del
  modelo. Se justifican por política de riesgo, no por su poder separador en
  la muestra.

La edad máxima de 85 años ilustra la diferencia: como regla de calidad
convertiría en nulo el dato de un cliente de 88 años y lo imputaría con la
mediana, inventando una edad falsa. Como regla de negocio, el dato se conserva
y el cliente se rechaza con una razón auditable.

---

## Estado del proyecto

- [x] Carga de datos
- [x] Análisis exploratorio
- [x] Feature engineering
- [ ] Modelo heurístico
- [ ] Entrenamiento y comparación de modelos
- [ ] Despliegue
- [ ] Evaluación
- [ ] Monitoreo y datadrift