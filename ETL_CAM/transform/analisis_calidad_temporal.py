import logging
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

INPUT_FILE = SCRIPT_DIR / "data" / "cam_transformado.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "calidad_temporal"
RESUMEN_FILE = OUTPUT_DIR / "resumen_calidad_temporal.csv"
HUECOS_FILE = OUTPUT_DIR / "huecos_temporales.csv"
LOG_FILE = SCRIPT_DIR / "analisis_calidad_temporal.log"

VARIABLE_PRINCIPAL = "nivel_m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AnalisisCalidadTemporalCAM:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_dir = OUTPUT_DIR
        self.resumen_file = RESUMEN_FILE
        self.huecos_file = HUECOS_FILE
        self.tiempo_inicio = time.time()

    def _detectar_frecuencia(self, serie_fechas: pd.Series):
        diferencias = serie_fechas.sort_values().diff().dropna()

        if diferencias.empty:
            return pd.NaT, None

        frecuencia = diferencias.mode().iloc[0]
        frecuencia_minutos = frecuencia.total_seconds() / 60

        return frecuencia, frecuencia_minutos

    def _clasificar_hueco(self, duracion_minutos: float) -> str:
        if duracion_minutos <= 60:
            return "corto"
        if duracion_minutos <= 24 * 60:
            return "medio"
        return "largo"

    def _analizar_estacion(self, estacion: str, df_estacion: pd.DataFrame):
        df_estacion = df_estacion.sort_values("fecha_hora").copy()
        fechas = df_estacion["fecha_hora"].dropna()

        if fechas.empty:
            return None, []

        fecha_minima = fechas.min()
        fecha_maxima = fechas.max()
        frecuencia, frecuencia_minutos = self._detectar_frecuencia(fechas)

        if frecuencia is pd.NaT or frecuencia_minutos is None or frecuencia_minutos <= 0:
            registros_esperados = len(fechas)
            porcentaje_cobertura = 100.0
        else:
            rango_esperado = pd.date_range(fecha_minima, fecha_maxima, freq=frecuencia)
            registros_esperados = len(rango_esperado)
            porcentaje_cobertura = (len(fechas.drop_duplicates()) / registros_esperados) * 100

        nivel_disponible = df_estacion[VARIABLE_PRINCIPAL].notna().sum()
        nivel_disponible_pct = (nivel_disponible / len(df_estacion)) * 100

        diferencias = fechas.sort_values().diff().dropna()
        huecos = []

        if frecuencia is not pd.NaT and frecuencia_minutos is not None:
            limite_hueco = frecuencia * 1.5
            fechas_ordenadas = fechas.sort_values().reset_index(drop=True)

            for i in range(1, len(fechas_ordenadas)):
                fecha_anterior = fechas_ordenadas.iloc[i - 1]
                fecha_actual = fechas_ordenadas.iloc[i]
                diferencia = fecha_actual - fecha_anterior

                if diferencia > limite_hueco:
                    duracion_minutos = diferencia.total_seconds() / 60
                    registros_faltantes_estimados = max(
                        int(round(duracion_minutos / frecuencia_minutos)) - 1,
                        1
                    )

                    huecos.append({
                        "estacion": estacion,
                        "inicio_hueco": fecha_anterior,
                        "fin_hueco": fecha_actual,
                        "duracion_minutos": round(duracion_minutos, 2),
                        "registros_faltantes_estimados": registros_faltantes_estimados,
                        "clasificacion_hueco": self._clasificar_hueco(duracion_minutos),
                    })

        hueco_maximo_minutos = (
            diferencias.max().total_seconds() / 60
            if not diferencias.empty
            else 0
        )

        resumen = {
            "estacion": estacion,
            "fecha_minima": fecha_minima,
            "fecha_maxima": fecha_maxima,
            "frecuencia_detectada": str(frecuencia),
            "frecuencia_minutos": round(frecuencia_minutos, 2) if frecuencia_minutos else None,
            "registros_reales": len(df_estacion),
            "registros_esperados": registros_esperados,
            "registros_faltantes_estimados": max(registros_esperados - len(fechas.drop_duplicates()), 0),
            "porcentaje_cobertura_temporal": round(porcentaje_cobertura, 2),
            "cantidad_huecos_detectados": len(huecos),
            "hueco_maximo_minutos": round(hueco_maximo_minutos, 2),
            "nivel_m_disponibles": int(nivel_disponible),
            "nivel_m_faltantes": int(df_estacion[VARIABLE_PRINCIPAL].isna().sum()),
            "nivel_m_disponible_pct": round(nivel_disponible_pct, 2),
            "apto_interpolacion_directa": len(huecos) > 0 and hueco_maximo_minutos <= 60,
        }

        return resumen, huecos

    def ejecutar(self) -> bool:
        if not self.input_file.exists():
            logger.error(f"No existe el archivo transformado: {self.input_file}")
            return False

        logger.info(f"Cargando datos transformados desde {self.input_file}")
        df = pd.read_csv(self.input_file, sep=";", encoding="utf-8-sig", low_memory=False)

        if "fecha_hora" not in df.columns or "estacion" not in df.columns:
            logger.error("El archivo debe contener las columnas fecha_hora y estacion")
            return False

        if VARIABLE_PRINCIPAL not in df.columns:
            logger.error(f"El archivo debe contener la variable principal {VARIABLE_PRINCIPAL}")
            return False

        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df = df.dropna(subset=["fecha_hora", "estacion"])

        resumenes = []
        huecos_totales = []

        for estacion, df_estacion in df.groupby("estacion"):
            logger.info(f"Analizando estacion: {estacion}")
            resumen, huecos = self._analizar_estacion(estacion, df_estacion)

            if resumen is not None:
                resumenes.append(resumen)
                huecos_totales.extend(huecos)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df_resumen = pd.DataFrame(resumenes)
        df_huecos = pd.DataFrame(huecos_totales)

        df_resumen.to_csv(self.resumen_file, index=False, sep=";", encoding="utf-8-sig")
        df_huecos.to_csv(self.huecos_file, index=False, sep=";", encoding="utf-8-sig")

        tiempo = round(time.time() - self.tiempo_inicio, 2)

        logger.info("Analisis de calidad temporal completado")
        logger.info(f"Resumen generado: {self.resumen_file}")
        logger.info(f"Huecos generados: {self.huecos_file}")
        logger.info(f"Estaciones analizadas: {len(resumenes)}")
        logger.info(f"Huecos detectados: {len(huecos_totales)}")
        logger.info(f"Tiempo ejecucion: {tiempo}s")

        return True


if __name__ == "__main__":
    analisis = AnalisisCalidadTemporalCAM()
    exito = analisis.ejecutar()
    exit(0 if exito else 1)
