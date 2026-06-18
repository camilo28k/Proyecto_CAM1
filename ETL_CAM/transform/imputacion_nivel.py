import logging
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "data" / "cam_transformado.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "imputacion"
OUTPUT_FILE = OUTPUT_DIR / "nivel_m_imputado.csv"
RESUMEN_FILE = OUTPUT_DIR / "resumen_imputacion_nivel.csv"
LOG_FILE = SCRIPT_DIR / "imputacion_nivel.log"

VARIABLE_OBJETIVO = "nivel_m"
FRECUENCIA_OBJETIVO = "5min"
LIMITE_INTERPOLACION_MINUTOS = 60
MIN_OBSERVACIONES_HISTORICAS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ImputadorNivelCAM:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_dir = OUTPUT_DIR
        self.output_file = OUTPUT_FILE
        self.resumen_file = RESUMEN_FILE
        self.tiempo_inicio = time.time()

    def _preparar_serie_estacion(self, df_estacion: pd.DataFrame) -> pd.DataFrame:
        df_estacion = df_estacion[["fecha_hora", "estacion", VARIABLE_OBJETIVO]].copy()
        df_estacion["fecha_hora"] = pd.to_datetime(df_estacion["fecha_hora"], errors="coerce")
        df_estacion[VARIABLE_OBJETIVO] = pd.to_numeric(df_estacion[VARIABLE_OBJETIVO], errors="coerce")
        df_estacion = df_estacion.dropna(subset=["fecha_hora", "estacion"])

        # Las estaciones no siempre registran exactamente en el minuto 00, 05, 10...
        # Por eso se redondea a la grilla de 5 minutos antes de imputar.
        df_estacion["fecha_hora"] = df_estacion["fecha_hora"].dt.round(FRECUENCIA_OBJETIVO)

        df_estacion = (
            df_estacion
            .sort_values("fecha_hora")
            .groupby(["estacion", "fecha_hora"], as_index=False)
            .agg({VARIABLE_OBJETIVO: "first"})
        )

        return df_estacion

    def _crear_grilla_temporal(self, df_estacion: pd.DataFrame, estacion: str) -> pd.DataFrame:
        fecha_minima = df_estacion["fecha_hora"].min()
        fecha_maxima = df_estacion["fecha_hora"].max()
        rango_fechas = pd.date_range(fecha_minima, fecha_maxima, freq=FRECUENCIA_OBJETIVO)

        df_grilla = pd.DataFrame({
            "fecha_hora": rango_fechas,
            "estacion": estacion,
        })

        df_grilla = df_grilla.merge(df_estacion, on=["estacion", "fecha_hora"], how="left")
        df_grilla = df_grilla.rename(columns={VARIABLE_OBJETIVO: "nivel_m_original"})

        return df_grilla

    def _aplicar_interpolacion_temporal(self, df_grilla: pd.DataFrame) -> pd.DataFrame:
        limite_periodos = int(LIMITE_INTERPOLACION_MINUTOS / 5)
        original_faltante = df_grilla["nivel_m_original"].isna()

        df_grilla["nivel_m_imputado"] = df_grilla["nivel_m_original"].interpolate(
            method="linear",
            limit=limite_periodos,
            limit_area="inside",
        )

        imputado_interpolacion = original_faltante & df_grilla["nivel_m_imputado"].notna()
        df_grilla.loc[imputado_interpolacion, "metodo_imputacion_nivel"] = "interpolacion_temporal_60min"
        df_grilla.loc[imputado_interpolacion, "calidad_imputacion_nivel"] = "alta"

        return df_grilla

    def _aplicar_promedio_historico(self, df_grilla: pd.DataFrame) -> pd.DataFrame:
        observados = df_grilla[df_grilla["nivel_m_original"].notna()].copy()

        if observados.empty:
            return df_grilla

        for df_aux in [df_grilla, observados]:
            df_aux["mes"] = df_aux["fecha_hora"].dt.month
            df_aux["dia"] = df_aux["fecha_hora"].dt.day
            df_aux["hora"] = df_aux["fecha_hora"].dt.hour
            df_aux["minuto"] = df_aux["fecha_hora"].dt.minute

        calendario = (
            observados
            .groupby(["mes", "dia", "hora", "minuto"])["nivel_m_original"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "promedio_calendario", "count": "n_calendario"})
        )

        mensual_horario = (
            observados
            .groupby(["mes", "hora", "minuto"])["nivel_m_original"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "promedio_mensual_horario", "count": "n_mensual_horario"})
        )

        df_grilla = df_grilla.merge(calendario, on=["mes", "dia", "hora", "minuto"], how="left")
        df_grilla = df_grilla.merge(mensual_horario, on=["mes", "hora", "minuto"], how="left")

        faltante = df_grilla["nivel_m_imputado"].isna()
        con_calendario = faltante & (df_grilla["n_calendario"] >= MIN_OBSERVACIONES_HISTORICAS)
        df_grilla.loc[con_calendario, "nivel_m_imputado"] = df_grilla.loc[con_calendario, "promedio_calendario"]
        df_grilla.loc[con_calendario, "metodo_imputacion_nivel"] = "promedio_historico_mes_dia_hora_minuto"
        df_grilla.loc[con_calendario, "calidad_imputacion_nivel"] = "media"

        faltante = df_grilla["nivel_m_imputado"].isna()
        con_mensual = faltante & (df_grilla["n_mensual_horario"] >= MIN_OBSERVACIONES_HISTORICAS)
        df_grilla.loc[con_mensual, "nivel_m_imputado"] = df_grilla.loc[con_mensual, "promedio_mensual_horario"]
        df_grilla.loc[con_mensual, "metodo_imputacion_nivel"] = "promedio_historico_mes_hora_minuto"
        df_grilla.loc[con_mensual, "calidad_imputacion_nivel"] = "baja"

        columnas_auxiliares = [
            "mes",
            "dia",
            "hora",
            "minuto",
            "promedio_calendario",
            "n_calendario",
            "promedio_mensual_horario",
            "n_mensual_horario",
        ]

        return df_grilla.drop(columns=columnas_auxiliares)

    def _imputar_estacion(self, estacion: str, df_estacion: pd.DataFrame):
        df_estacion = self._preparar_serie_estacion(df_estacion)

        if df_estacion[VARIABLE_OBJETIVO].notna().sum() == 0:
            logger.warning(f"Estacion omitida sin datos de {VARIABLE_OBJETIVO}: {estacion}")
            return None, {
                "estacion": estacion,
                "estado": "omitida_sin_nivel_m",
                "registros_generados": 0,
                "nivel_m_observados": 0,
                "nivel_m_faltantes_originales": len(df_estacion),
                "imputados_interpolacion": 0,
                "imputados_historico_calendario": 0,
                "imputados_historico_mensual": 0,
                "sin_imputar": len(df_estacion),
            }

        df_grilla = self._crear_grilla_temporal(df_estacion, estacion)

        df_grilla["nivel_m_imputado"] = df_grilla["nivel_m_original"]
        df_grilla["metodo_imputacion_nivel"] = "observado"
        df_grilla["calidad_imputacion_nivel"] = "real"

        faltantes_originales = df_grilla["nivel_m_original"].isna()
        df_grilla.loc[faltantes_originales, "metodo_imputacion_nivel"] = "sin_imputar"
        df_grilla.loc[faltantes_originales, "calidad_imputacion_nivel"] = "sin_dato"

        df_grilla = self._aplicar_interpolacion_temporal(df_grilla)
        df_grilla = self._aplicar_promedio_historico(df_grilla)

        sin_imputar = df_grilla["nivel_m_imputado"].isna()
        df_grilla.loc[sin_imputar, "metodo_imputacion_nivel"] = "sin_imputar"
        df_grilla.loc[sin_imputar, "calidad_imputacion_nivel"] = "sin_dato"

        df_grilla["fue_imputado_nivel"] = (
            df_grilla["nivel_m_original"].isna()
            & df_grilla["nivel_m_imputado"].notna()
        )

        resumen = {
            "estacion": estacion,
            "estado": "procesada",
            "fecha_minima": df_grilla["fecha_hora"].min(),
            "fecha_maxima": df_grilla["fecha_hora"].max(),
            "registros_generados": len(df_grilla),
            "nivel_m_observados": int(df_grilla["nivel_m_original"].notna().sum()),
            "nivel_m_faltantes_originales": int(df_grilla["nivel_m_original"].isna().sum()),
            "imputados_interpolacion": int((df_grilla["metodo_imputacion_nivel"] == "interpolacion_temporal_60min").sum()),
            "imputados_historico_calendario": int((df_grilla["metodo_imputacion_nivel"] == "promedio_historico_mes_dia_hora_minuto").sum()),
            "imputados_historico_mensual": int((df_grilla["metodo_imputacion_nivel"] == "promedio_historico_mes_hora_minuto").sum()),
            "sin_imputar": int(df_grilla["nivel_m_imputado"].isna().sum()),
        }

        columnas_salida = [
            "fecha_hora",
            "estacion",
            "nivel_m_original",
            "nivel_m_imputado",
            "fue_imputado_nivel",
            "metodo_imputacion_nivel",
            "calidad_imputacion_nivel",
        ]

        return df_grilla[columnas_salida], resumen

    def ejecutar(self) -> bool:
        if not self.input_file.exists():
            logger.error(f"No existe el archivo transformado: {self.input_file}")
            return False

        logger.info(f"Cargando datos transformados desde {self.input_file}")
        df = pd.read_csv(
            self.input_file,
            sep=";",
            encoding="utf-8-sig",
            usecols=["fecha_hora", "estacion", VARIABLE_OBJETIVO],
            low_memory=False,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.output_file.exists():
            self.output_file.unlink()

        resumenes = []
        primera_escritura = True

        for estacion, df_estacion in df.groupby("estacion"):
            logger.info(f"Imputando estacion: {estacion}")
            df_imputado, resumen = self._imputar_estacion(estacion, df_estacion)
            resumenes.append(resumen)

            if df_imputado is None:
                continue

            df_imputado.to_csv(
                self.output_file,
                index=False,
                sep=";",
                encoding="utf-8-sig",
                mode="w" if primera_escritura else "a",
                header=primera_escritura,
            )
            primera_escritura = False

        pd.DataFrame(resumenes).to_csv(
            self.resumen_file,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

        tiempo = round(time.time() - self.tiempo_inicio, 2)

        logger.info("Imputacion de nivel completada")
        logger.info(f"Archivo generado: {self.output_file}")
        logger.info(f"Resumen generado: {self.resumen_file}")
        logger.info(f"Tiempo ejecucion: {tiempo}s")

        return True


if __name__ == "__main__":
    imputador = ImputadorNivelCAM()
    exito = imputador.ejecutar()
    exit(0 if exito else 1)
