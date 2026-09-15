"""
app.py

Interfaz de disponibilizacion del modelo de riesgo crediticio.

Permite evaluar solicitudes de dos formas: cargando un archivo con un lote
completo, o diligenciando una solicitud individual. La decision proviene de
model_deploy, de modo que ambas rutas aplican exactamente la misma logica que
correria en produccion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from ft_engineering import cargar_config  # noqa: E402
from model_deploy import (  # noqa: E402
    APROBADO,
    RECHAZADO_POR_MODELO,
    RECHAZADO_POR_REGLA,
    cargar_artefacto,
    decidir,
    resumir_decisiones,
)

st.set_page_config(page_title="Evaluacion de riesgo crediticio", layout="wide")


@st.cache_resource
def inicializar():
    """Carga configuracion y modelo una sola vez por sesion."""
    config = cargar_config()
    return config, cargar_artefacto()


config, artefacto = inicializar()
metadatos = artefacto["metadatos"]
umbral = config["decision"]["umbral_corte"]

st.title("Evaluacion de riesgo crediticio")
st.caption(
    f"Modelo: {metadatos['modelo']} · "
    f"Umbral de corte: {umbral} · "
    f"PR-AUC en evaluacion: {metadatos['metricas_test']['pr_auc']:.4f}"
)

with st.sidebar:
    st.subheader("Como decide el sistema")
    st.markdown(
        """
        1. Se aplican las **reglas duras** de politica de credito.
           Si alguna se activa, la solicitud se rechaza con el motivo
           correspondiente.
        2. Las solicitudes que ninguna regla marca pasan al **modelo
           estadistico**, que estima la probabilidad de incumplimiento.
        3. Si esa probabilidad alcanza el umbral, se rechaza.
        """
    )
    st.subheader("Reglas duras vigentes")
    for nombre, regla in config["reglas_heuristicas"].items():
        if regla["activa"]:
            st.markdown(f"**{nombre}** — {regla['descripcion']}")

lote, individual = st.tabs(["Evaluacion por lote", "Solicitud individual"])


def mostrar_resultados(decisiones: pd.DataFrame, entrada: pd.DataFrame) -> None:
    """Presenta el resumen y el detalle de un lote evaluado."""
    resumen = resumir_decisiones(decisiones)

    columnas = st.columns(4)
    conteos = decisiones["decision"].value_counts()
    columnas[0].metric("Solicitudes", len(decisiones))
    columnas[1].metric("Aprobadas", int(conteos.get(APROBADO, 0)))
    columnas[2].metric("Rechazadas por regla",
                       int(conteos.get(RECHAZADO_POR_REGLA, 0)))
    columnas[3].metric("Rechazadas por modelo",
                       int(conteos.get(RECHAZADO_POR_MODELO, 0)))

    st.dataframe(resumen, hide_index=True, width="stretch")

    detalle = pd.concat([entrada.reset_index(drop=True), decisiones], axis=1)
    st.subheader("Detalle por solicitud")
    st.dataframe(detalle, width="stretch")

    st.download_button(
        "Descargar resultados",
        detalle.to_csv(index=False).encode("utf-8"),
        file_name="decisiones_crediticias.csv",
        mime="text/csv",
    )


with lote:
    st.markdown(
        "Cargue un archivo con las mismas columnas del dataset original. "
        "Cada solicitud recibe una decision, su motivo y la probabilidad "
        "estimada de incumplimiento."
    )
    archivo = st.file_uploader("Archivo de solicitudes (CSV)", type=["csv"])
    separador = st.selectbox("Separador de columnas", [";", ",", "|"])

    if archivo is not None:
        entrada = pd.read_csv(archivo, sep=separador)
        st.success(f"Archivo cargado: {len(entrada)} solicitudes")

        if st.button("Evaluar lote", type="primary"):
            with st.spinner("Evaluando solicitudes..."):
                decisiones = decidir(entrada, config, artefacto)
            mostrar_resultados(decisiones, entrada)


with individual:
    st.markdown(
        "Diligencie los datos de una solicitud para ver la decision y su "
        "motivo."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Credito**")
        tipo_credito = st.selectbox("Tipo de credito", [4, 9])
        capital_prestado = st.number_input(
            "Capital solicitado", min_value=100_000, value=1_500_000, step=100_000)
        plazo_meses = st.number_input(
            "Plazo (meses)", min_value=1, max_value=180, value=12)
        cuota_pactada = st.number_input(
            "Cuota pactada", min_value=1, value=150_000, step=10_000)

    with col2:
        st.markdown("**Cliente**")
        edad_cliente = st.number_input("Edad", min_value=18, max_value=100, value=35)
        tipo_laboral = st.selectbox("Vinculacion", ["Empleado", "Independiente"])
        salario_cliente = st.number_input(
            "Ingreso mensual", min_value=0, value=3_000_000, step=100_000)
        total_otros_prestamos = st.number_input(
            "Otras obligaciones", min_value=0, value=1_000_000, step=100_000)

    with col3:
        st.markdown("**Centrales de riesgo**")
        puntaje_datacredito = st.slider("Puntaje de central", 150, 950, 700)
        saldo_total = st.number_input("Saldo reportado", min_value=0, value=20_000)
        saldo_mora = st.number_input("Saldo en mora", min_value=0, value=0)
        huella_consulta = st.number_input(
            "Consultas recientes", min_value=0, value=2)
        cant_creditosvigentes = st.number_input(
            "Creditos vigentes", min_value=0, value=2)

    with st.expander("Obligaciones por sector"):
        sec1, sec2, sec3 = st.columns(3)
        creditos_financiero = sec1.number_input("Sector financiero", 0, value=2)
        creditos_cooperativo = sec2.number_input("Sector cooperativo", 0, value=0)
        creditos_real = sec3.number_input("Sector real", 0, value=0)

    if st.button("Evaluar solicitud", type="primary"):
        solicitud = pd.DataFrame([{
            "tipo_credito": tipo_credito,
            "fecha_prestamo": "01/01/2026",
            "capital_prestado": capital_prestado,
            "plazo_meses": plazo_meses,
            "edad_cliente": edad_cliente,
            "tipo_laboral": tipo_laboral,
            "salario_cliente": salario_cliente,
            "total_otros_prestamos": total_otros_prestamos,
            "cuota_pactada": cuota_pactada,
            "puntaje": 0,
            "puntaje_datacredito": puntaje_datacredito,
            "cant_creditosvigentes": cant_creditosvigentes,
            "huella_consulta": huella_consulta,
            "saldo_mora": saldo_mora,
            "saldo_total": saldo_total,
            "saldo_principal": saldo_total,
            "saldo_mora_codeudor": 0,
            "creditos_sectorFinanciero": creditos_financiero,
            "creditos_sectorCooperativo": creditos_cooperativo,
            "creditos_sectorReal": creditos_real,
            "promedio_ingresos_datacredito": salario_cliente,
            "tendencia_ingresos": "Estable",
        }])

        decision = decidir(solicitud, config, artefacto).iloc[0]

        if decision["decision"] == APROBADO:
            st.success(f"**{decision['decision']}**")
        else:
            st.error(f"**{decision['decision']}**")

        st.write(f"**Motivo:** {decision['motivo']}")

        probabilidad = decision["probabilidad_incumplimiento"]
        if pd.notna(probabilidad):
            st.metric("Probabilidad estimada de incumplimiento",
                      f"{float(probabilidad):.2%}")
            st.progress(min(float(probabilidad), 1.0))
            st.caption(f"Umbral de rechazo: {umbral:.0%}")
        else:
            st.info(
                "La solicitud fue resuelta por una regla dura, que prevalece "
                "sobre la prediccion del modelo. No se calculo probabilidad."
            )