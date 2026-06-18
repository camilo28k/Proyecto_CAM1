import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "data" / "cam_transformado.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "imputacion_general"
OUTPUT_FILE = OUTPUT_DIR / "variables_modelo_imputadas.csv"
RESUMEN_FILE = OUTPUT_DIR / "resumen_imputacion_general.csv"
CALIDAD_FILE = OUTPUT_DIR / "calidad_variables_por_estacion.csv"
LOG_FILE = SCRIPT_DIR / "imputacion_general.log"

FRECUENCIA_OBJETIVO = "5min"
LIMITE_INTERPOLACION_MINUTOS = 60
MIN_OBSERVACIONES_HISTORICAS = 2

VARIABLES_MODELO = [
    "nivel_m",
    "temperatura_c",
    "humedad_pct",
    "radiacion_solar_w_m2",
    "lluvia_mm",
    "direccion_viento_grados",
    "velocidad_viento_m_s",
    "presion_atmosferica_hpa",
    "voltaje_bateria_vdc",
    "velocidad_agua_m_s",
]

VARIABLES_SIN_INTERPOLACION_LINEAL = {
    "lluvia_mm",
}

VARIABLE_DIRECCION = "direccion_viento_grados"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ImputadorGeneralCAM:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_dir = OUTPUT_DIR
        self.output_file = OUTPUT_FILE
        self.resumen_file = RESUMEN_FILE
        self.calidad_file = CALIDAD_FILE
        self.tiempo_inicio = time.time()

    def _crear_grilla_estacion(self, df_estacion: pd.DataFrame, estacion: str) -> pd.DataFrame:
        df_estacion = df_estacion.copy()
        df_estacion["fecha_hora"] = pd.to_datetime(df_estacion["fecha_hora"], errors="coerce")
        df_estacion = df_estacion.dropna(subset=["fecha_hora", "estacion"])
        df_estacion["fecha_hora"] = df_estacion["fecha_hora"].dt.round(FRECUENCIA_OBJETIVO)

        for variable in VARIABLES_MODELO:
            df_estacion[variable] = pd.to_numeric(df_estacion[variable], errors="coerce")

        df_estacion = (
            df_estacion
            .sort_values("fecha_hora")
            .groupby(["estacion", "fecha_hora"], as_index=False)[VARIABLES_MODELO]
            .first()
        )

        fecha_minima = df_estacion["fecha_hora"].min()
        fecha_maxima = df_estacion["fecha_hora"].max()
        rango_fechas = pd.date_range(fecha_minima, fecha_maxima, freq=FRECUENCIA_OBJETIVO)

        df_grilla = pd.DataFrame({
            "fecha_hora": rango_fechas,
            "estacion": estacion,
        })

        return df_grilla.merge(df_estacion, on=["estacion", "fecha_hora"], how="left")

    def _agregar_calendario(self, df: pd.DataFrame) -> pd.DataFrame:
        df["mes"] = df["fecha_hora"].dt.month
        df["dia"] = df["fecha_hora"].dt.day
        df["hora"] = df["fecha_hora"].dt.hour
        df["minuto"] = df["fecha_hora"].dt.minute
        return df

    def _imputar_variable_numerica(self, df: pd.DataFrame, variable: str):
        original = df[variable].copy()
        valor_final = original.copy()
        metodo = pd.Series("observado", index=df.index, dtype="string")
        calidad = pd.Series("real", index=df.index, dtype="string")

        faltante_original = original.isna()
        metodo.loc[faltante_original] = "sin_imputar"
        calidad.loc[faltante_original] = "sin_dato"

        if original.notna().sum() == 0:
            return valor_final, metodo, calidad

        if variable not in VARIABLES_SIN_INTERPOLACION_LINEAL:
            limite_periodos = int(LIMITE_INTERPOLACION_MINUTOS / 5)
            interpolado = valor_final.interpolate(
                method="linear",
                limit=limite_periodos,
                limit_area="inside",
            )
            mascara_interpolada = faltante_original & interpolado.notna()
            valor_final.loc[mascara_interpolada] = interpolado.loc[mascara_interpolada]
            metodo.loc[mascara_interpolada] = "interpolacion_temporal_60min"
            calidad.loc[mascara_interpolada] = "alta"

        df_aux = self._agregar_calendario(pd.DataFrame({
            "fecha_hora": df["fecha_hora"],
            "valor": original,
        }))
        observados = df_aux[df_aux["valor"].notna()].copy()

        calendario = (
            observados
            .groupby(["mes", "dia", "hora", "minuto"])["valor"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "promedio_calendario", "count": "n_calendario"})
        )

        mensual_horario = (
            observados
            .groupby(["mes", "hora", "minuto"])["valor"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "promedio_mensual", "count": "n_mensual"})
        )

        df_aux = df_aux.merge(calendario, on=["mes", "dia", "hora", "minuto"], how="left")
        df_aux = df_aux.merge(mensual_horario, on=["mes", "hora", "minuto"], how="left")

        faltante = valor_final.isna()
        con_calendario = faltante & (df_aux["n_calendario"] >= MIN_OBSERVACIONES_HISTORICAS)
        valor_final.loc[con_calendario] = df_aux.loc[con_calendario, "promedio_calendario"]
        metodo.loc[con_calendario] = "promedio_historico_mes_dia_hora_minuto"
        calidad.loc[con_calendario] = "media"

        faltante = valor_final.isna()
        con_mensual = faltante & (df_aux["n_mensual"] >= MIN_OBSERVACIONES_HISTORICAS)
        valor_final.loc[con_mensual] = df_aux.loc[con_mensual, "promedio_mensual"]
        metodo.loc[con_mensual] = "promedio_historico_mes_hora_minuto"
        calidad.loc[con_mensual] = "baja"

        return valor_final, metodo, calidad

    def _media_circular_grados(self, valores: pd.Series):
        valores = pd.to_numeric(valores, errors="coerce").dropna()

        if valores.empty:
            return pd.NA

        radianes = valores.apply(math.radians)
        seno = radianes.apply(math.sin).mean()
        coseno = radianes.apply(math.cos).mean()
        angulo = math.degrees(math.atan2(seno, coseno))

        return (angulo + 360) % 360

    def _imputar_direccion_viento(self, df: pd.DataFrame):
        variable = VARIABLE_DIRECCION
        original = df[variable].copy()
        valor_final = original.copy()
        metodo = pd.Series("observado", index=df.index, dtype="string")
        calidad = pd.Series("real", index=df.index, dtype="string")

        faltante_original = original.isna()
        metodo.loc[faltante_original] = "sin_imputar"
        calidad.loc[faltante_original] = "sin_dato"

        if original.notna().sum() == 0:
            return valor_final, metodo, calidad

        radianes = np.deg2rad(original.astype("float"))
        seno = pd.Series(np.sin(radianes), index=df.index, dtype="float")
        coseno = pd.Series(np.cos(radianes), index=df.index, dtype="float")

        limite_periodos = int(LIMITE_INTERPOLACION_MINUTOS / 5)
        seno_interp = seno.interpolate(method="linear", limit=limite_periodos, limit_area="inside")
        coseno_interp = coseno.interpolate(method="linear", limit=limite_periodos, limit_area="inside")
        mascara_interpolada = faltante_original & seno_interp.notna() & coseno_interp.notna()

        angulos_interp = (
            pd.Series(np.rad2deg(np.arctan2(seno_interp, coseno_interp)), index=df.index)
            + 360
        ) % 360

        valor_final.loc[mascara_interpolada] = angulos_interp.loc[mascara_interpolada]
        metodo.loc[mascara_interpolada] = "interpolacion_circular_60min"
        calidad.loc[mascara_interpolada] = "alta"

        df_aux = self._agregar_calendario(pd.DataFrame({
            "fecha_hora": df["fecha_hora"],
            "valor": original,
        }))
        observados = df_aux[df_aux["valor"].notna()].copy()

        calendario = (
            observados
            .groupby(["mes", "dia", "hora", "minuto"])["valor"]
            .agg(promedio_circular=self._media_circular_grados, n_calendario="count")
            .reset_index()
        )

        mensual = (
            observados
            .groupby(["mes", "hora", "minuto"])["valor"]
            .agg(promedio_circular_mes_hora_minuto=self._media_circular_grados, n_mensual="count")
            .reset_index()
        )

        df_aux = df_aux.merge(calendario, on=["mes", "dia", "hora", "minuto"], how="left")
        df_aux = df_aux.merge(mensual, on=["mes", "hora", "minuto"], how="left")

        faltante = valor_final.isna()
        con_calendario = faltante & (df_aux["n_calendario"] >= MIN_OBSERVACIONES_HISTORICAS)
        valor_final.loc[con_calendario] = df_aux.loc[con_calendario, "promedio_circular"]
        metodo.loc[con_calendario] = "promedio_circular_mes_dia_hora_minuto"
        calidad.loc[con_calendario] = "media"

        faltante = valor_final.isna()
        con_mensual = faltante & (df_aux["n_mensual"] >= MIN_OBSERVACIONES_HISTORICAS)
        valor_final.loc[con_mensual] = df_aux.loc[con_mensual, "promedio_circular_mes_hora_minuto"]
        metodo.loc[con_mensual] = "promedio_circular_mes_hora_minuto"
        calidad.loc[con_mensual] = "baja"

        return valor_final, metodo, calidad

    def _resumen_variable(self, estacion: str, variable: str, original: pd.Series, valor_final: pd.Series, metodo: pd.Series):
        total = len(valor_final)
        observados = int(original.notna().sum())
        faltantes_originales = int(original.isna().sum())
        sin_imputar = int(valor_final.isna().sum())

        return {
            "estacion": estacion,
            "variable": variable,
            "registros": total,
            "observados": observados,
            "faltantes_originales": faltantes_originales,
            "imputados_interpolacion": int(metodo.str.contains("interpolacion", na=False).sum()),
            "imputados_historico_calendario": int(metodo.str.contains("mes_dia_hora_minuto", na=False).sum()),
            "imputados_historico_mensual": int(metodo.str.contains("mes_hora_minuto", na=False).sum()),
            "sin_imputar": sin_imputar,
            "observados_pct": round((observados / total) * 100, 2) if total else 0,
            "imputados_pct": round(((faltantes_originales - sin_imputar) / total) * 100, 2) if total else 0,
            "sin_imputar_pct": round((sin_imputar / total) * 100, 2) if total else 0,
        }

    def _procesar_estacion(self, estacion: str, df_estacion: pd.DataFrame):
        logger.info(f"Imputando variables para estacion: {estacion}")
        df_grilla = self._crear_grilla_estacion(df_estacion, estacion)

        salida = df_grilla[["fecha_hora", "estacion"]].copy()
        resumenes = []

        for variable in VARIABLES_MODELO:
            original = df_grilla[variable].copy()

            if variable == VARIABLE_DIRECCION:
                valor_final, metodo, calidad = self._imputar_direccion_viento(df_grilla)
            else:
                valor_final, metodo, calidad = self._imputar_variable_numerica(df_grilla, variable)

            salida[variable] = valor_final
            salida[f"{variable}_fue_imputado"] = original.isna() & valor_final.notna()
            salida[f"{variable}_calidad"] = calidad
            resumenes.append(self._resumen_variable(estacion, variable, original, valor_final, metodo))

        salida["variables_disponibles"] = salida[VARIABLES_MODELO].notna().sum(axis=1)
        salida["variables_faltantes"] = salida[VARIABLES_MODELO].isna().sum(axis=1)

        return salida, resumenes

    def ejecutar(self) -> bool:
        if not self.input_file.exists():
            logger.error(f"No existe el archivo transformado: {self.input_file}")
            return False

        logger.info(f"Cargando datos transformados desde {self.input_file}")
        columnas = ["fecha_hora", "estacion", *VARIABLES_MODELO]
        df = pd.read_csv(
            self.input_file,
            sep=";",
            encoding="utf-8-sig",
            usecols=columnas,
            low_memory=False,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.output_file.exists():
            self.output_file.unlink()

        primera_escritura = True
        resumenes = []

        for estacion, df_estacion in df.groupby("estacion"):
            df_salida, resumen_estacion = self._procesar_estacion(estacion, df_estacion)
            resumenes.extend(resumen_estacion)

            df_salida.to_csv(
                self.output_file,
                index=False,
                sep=";",
                encoding="utf-8-sig",
                mode="w" if primera_escritura else "a",
                header=primera_escritura,
            )
            primera_escritura = False

        df_resumen = pd.DataFrame(resumenes)
        df_resumen.to_csv(self.resumen_file, index=False, sep=";", encoding="utf-8-sig")

        df_calidad = df_resumen.pivot(
            index="estacion",
            columns="variable",
            values="observados_pct",
        ).reset_index()
        df_calidad.to_csv(self.calidad_file, index=False, sep=";", encoding="utf-8-sig")

        tiempo = round(time.time() - self.tiempo_inicio, 2)

        logger.info("Imputacion general completada")
        logger.info(f"Archivo generado: {self.output_file}")
        logger.info(f"Resumen generado: {self.resumen_file}")
        logger.info(f"Calidad generada: {self.calidad_file}")
        logger.info(f"Tiempo ejecucion: {tiempo}s")

        return True


if __name__ == "__main__":
    imputador = ImputadorGeneralCAM()
    exito = imputador.ejecutar()
    exit(0 if exito else 1)
