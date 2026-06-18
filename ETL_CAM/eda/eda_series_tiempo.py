import logging
import html
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

INPUT_FILE = PROJECT_DIR / "ETL_CAM" / "transform" / "data" / "imputacion_general" / "variables_modelo_imputadas.csv"
OUTPUT_DIR = SCRIPT_DIR / "reportes"
GRAFICAS_DIR = SCRIPT_DIR / "graficas"
LOG_FILE = SCRIPT_DIR / "eda_series_tiempo.log"

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

VARIABLE_OBJETIVO = "nivel_m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

COLORES = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#0f766e",
]


class EDASeriesTiempoCAM:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_dir = OUTPUT_DIR
        self.graficas_dir = GRAFICAS_DIR

    def _cargar_datos(self) -> pd.DataFrame:
        if not self.input_file.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {self.input_file}")

        logger.info(f"Cargando datos desde {self.input_file}")
        columnas_necesarias = ["fecha_hora", "estacion", *VARIABLES_MODELO]

        for variable in VARIABLES_MODELO:
            columnas_necesarias.extend([
                f"{variable}_fue_imputado",
                f"{variable}_calidad",
            ])

        df = pd.read_csv(
            self.input_file,
            sep=";",
            encoding="utf-8-sig",
            usecols=lambda col: col in columnas_necesarias,
            low_memory=False,
        )

        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df = df.dropna(subset=["fecha_hora", "estacion"])

        for variable in VARIABLES_MODELO:
            if variable in df.columns:
                df[variable] = pd.to_numeric(df[variable], errors="coerce")

        df["anio"] = df["fecha_hora"].dt.year
        df["mes"] = df["fecha_hora"].dt.month
        df["hora"] = df["fecha_hora"].dt.hour
        df["fecha"] = df["fecha_hora"].dt.date

        return df

    def _guardar_resumen_general(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        resumen = {
            "registros": len(df),
            "estaciones": df["estacion"].nunique(),
            "fecha_minima": df["fecha_hora"].min(),
            "fecha_maxima": df["fecha_hora"].max(),
            "variables_modelo": len(VARIABLES_MODELO),
        }

        pd.DataFrame([resumen]).to_csv(
            self.output_dir / "resumen_eda_general.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    def _guardar_resumen_estaciones(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        resumen = (
            df.groupby("estacion")
            .agg(
                registros=("fecha_hora", "count"),
                fecha_minima=("fecha_hora", "min"),
                fecha_maxima=("fecha_hora", "max"),
                nivel_m_promedio=(VARIABLE_OBJETIVO, "mean"),
                nivel_m_minimo=(VARIABLE_OBJETIVO, "min"),
                nivel_m_maximo=(VARIABLE_OBJETIVO, "max"),
                nivel_m_desviacion=(VARIABLE_OBJETIVO, "std"),
            )
            .reset_index()
        )

        resumen.to_csv(
            self.output_dir / "resumen_eda_estaciones.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    def _guardar_resumen_variables(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filas = []

        for estacion, df_estacion in df.groupby("estacion"):
            total = len(df_estacion)
            for variable in VARIABLES_MODELO:
                if variable not in df_estacion.columns:
                    continue

                imputado_col = f"{variable}_fue_imputado"
                calidad_col = f"{variable}_calidad"
                observados = df_estacion[variable].notna().sum()

                fila = {
                    "estacion": estacion,
                    "variable": variable,
                    "registros": total,
                    "disponibles": int(observados),
                    "faltantes": int(total - observados),
                    "disponible_pct": round((observados / total) * 100, 2) if total else 0,
                    "media": df_estacion[variable].mean(),
                    "minimo": df_estacion[variable].min(),
                    "maximo": df_estacion[variable].max(),
                    "desviacion": df_estacion[variable].std(),
                }

                if imputado_col in df_estacion.columns:
                    imputados = df_estacion[imputado_col].astype("string").str.lower().eq("true").sum()
                    fila["imputados"] = int(imputados)
                    fila["imputados_pct"] = round((imputados / total) * 100, 2) if total else 0

                if calidad_col in df_estacion.columns:
                    conteo_calidad = df_estacion[calidad_col].value_counts(dropna=False).to_dict()
                    for calidad, cantidad in conteo_calidad.items():
                        fila[f"calidad_{calidad}"] = int(cantidad)

                filas.append(fila)

        pd.DataFrame(filas).to_csv(
            self.output_dir / "resumen_variables.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    def _guardar_eventos_extremos(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        eventos = []

        for estacion, df_estacion in df.groupby("estacion"):
            serie = df_estacion[VARIABLE_OBJETIVO].dropna()
            if serie.empty:
                continue

            umbral_p95 = serie.quantile(0.95)
            df_eventos = df_estacion[df_estacion[VARIABLE_OBJETIVO] >= umbral_p95].copy()
            df_eventos["umbral_p95_estacion"] = umbral_p95
            eventos.append(df_eventos[[
                "fecha_hora",
                "estacion",
                VARIABLE_OBJETIVO,
                "umbral_p95_estacion",
            ]])

        if eventos:
            df_eventos = pd.concat(eventos, ignore_index=True)
        else:
            df_eventos = pd.DataFrame(columns=["fecha_hora", "estacion", VARIABLE_OBJETIVO, "umbral_p95_estacion"])

        df_eventos.to_csv(
            self.output_dir / "eventos_nivel_extremo.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    def _guardar_correlacion(self, df: pd.DataFrame) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        correlacion = df[VARIABLES_MODELO].corr(numeric_only=True)
        correlacion.to_csv(
            self.output_dir / "correlacion_variables.csv",
            sep=";",
            encoding="utf-8-sig",
        )
        return correlacion

    def _escala(self, valor, minimo, maximo, salida_min, salida_max):
        if pd.isna(valor) or maximo == minimo:
            return (salida_min + salida_max) / 2
        return salida_min + ((valor - minimo) / (maximo - minimo)) * (salida_max - salida_min)

    def _guardar_svg(self, nombre_archivo: str, contenido: str, ancho: int = 1200, alto: int = 700) -> None:
        self.graficas_dir.mkdir(parents=True, exist_ok=True)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" '
            f'viewBox="0 0 {ancho} {alto}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f"{contenido}\n"
            "</svg>\n"
        )
        (self.graficas_dir / nombre_archivo).write_text(svg, encoding="utf-8")

    def _texto(self, x, y, texto, tamano=14, ancla="middle", peso="normal", color="#111827"):
        texto = html.escape(str(texto))
        return (
            f'<text x="{x}" y="{y}" font-family="Arial" font-size="{tamano}" '
            f'font-weight="{peso}" text-anchor="{ancla}" fill="{color}">{texto}</text>'
        )

    def _ejes(self, margen_izq, margen_sup, ancho_plot, alto_plot):
        x0 = margen_izq
        y0 = margen_sup + alto_plot
        x1 = margen_izq + ancho_plot
        y1 = margen_sup
        lineas = [
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#374151" stroke-width="1"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#374151" stroke-width="1"/>',
        ]
        for i in range(1, 5):
            y = margen_sup + (alto_plot / 5) * i
            lineas.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        return "\n".join(lineas)

    def _grafica_lineas(self, datos_por_etiqueta, titulo, xlabel, ylabel, archivo, x_texto=False):
        ancho = 1200
        alto = 700
        margen_izq = 90
        margen_sup = 70
        ancho_plot = 900
        alto_plot = 500

        todos_x = []
        todos_y = []
        for datos in datos_por_etiqueta.values():
            for x, y in datos:
                if pd.notna(x) and pd.notna(y):
                    todos_x.append(x)
                    todos_y.append(y)

        if not todos_x or not todos_y:
            self._guardar_svg(archivo, self._texto(600, 350, f"Sin datos para {titulo}", 20))
            return

        min_x, max_x = min(todos_x), max(todos_x)
        min_y, max_y = min(todos_y), max(todos_y)
        if min_y == max_y:
            min_y -= 1
            max_y += 1

        partes = [
            self._texto(600, 35, titulo, 22, peso="bold"),
            self._ejes(margen_izq, margen_sup, ancho_plot, alto_plot),
            self._texto(540, 660, xlabel, 14),
            self._texto(20, 330, ylabel, 14, ancla="middle"),
        ]

        for tick in range(6):
            y_val = min_y + ((max_y - min_y) / 5) * tick
            y_svg = self._escala(y_val, min_y, max_y, margen_sup + alto_plot, margen_sup)
            partes.append(self._texto(80, y_svg + 4, f"{y_val:.2f}", 11, ancla="end", color="#4b5563"))

        for idx, (etiqueta, datos) in enumerate(datos_por_etiqueta.items()):
            puntos = []
            for x, y in datos:
                if pd.isna(x) or pd.isna(y):
                    continue
                px = self._escala(x, min_x, max_x, margen_izq, margen_izq + ancho_plot)
                py = self._escala(y, min_y, max_y, margen_sup + alto_plot, margen_sup)
                puntos.append(f"{px:.2f},{py:.2f}")

            if not puntos:
                continue

            color = COLORES[idx % len(COLORES)]
            partes.append(
                f'<polyline points="{" ".join(puntos)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            )
            leyenda_y = 90 + (idx * 22)
            partes.append(f'<rect x="1020" y="{leyenda_y - 10}" width="12" height="12" fill="{color}"/>')
            partes.append(self._texto(1040, leyenda_y, etiqueta, 12, ancla="start"))

        self._guardar_svg(archivo, "\n".join(partes), ancho, alto)

    def _grafica_nivel_por_estacion(self, df: pd.DataFrame) -> None:
        datos_por_estacion = {}

        for estacion, df_estacion in df.groupby("estacion"):
            if df_estacion[VARIABLE_OBJETIVO].notna().sum() == 0:
                continue

            serie_diaria = (
                df_estacion
                .set_index("fecha_hora")[VARIABLE_OBJETIVO]
                .resample("D")
                .mean()
            )
            datos_por_estacion[estacion] = [
                (fecha.toordinal(), valor)
                for fecha, valor in serie_diaria.items()
            ]

        self._grafica_lineas(
            datos_por_estacion,
            "Nivel del agua promedio diario por estacion",
            "Fecha",
            "Nivel (m)",
            "nivel_m_por_estacion.svg",
        )

    def _grafica_promedio_mensual(self, df: pd.DataFrame) -> None:
        mensual = (
            df.groupby(["estacion", "mes"])[VARIABLE_OBJETIVO]
            .mean()
            .reset_index()
        )

        datos_por_estacion = {}
        for estacion, datos in mensual.groupby("estacion"):
            datos_por_estacion[estacion] = list(zip(datos["mes"], datos[VARIABLE_OBJETIVO]))

        self._grafica_lineas(
            datos_por_estacion,
            "Promedio mensual de nivel_m",
            "Mes",
            "Nivel promedio (m)",
            "nivel_m_promedio_mensual.svg",
        )

    def _grafica_promedio_horario(self, df: pd.DataFrame) -> None:
        horario = (
            df.groupby(["estacion", "hora"])[VARIABLE_OBJETIVO]
            .mean()
            .reset_index()
        )

        datos_por_estacion = {}
        for estacion, datos in horario.groupby("estacion"):
            datos_por_estacion[estacion] = list(zip(datos["hora"], datos[VARIABLE_OBJETIVO]))

        self._grafica_lineas(
            datos_por_estacion,
            "Promedio horario de nivel_m",
            "Hora del dia",
            "Nivel promedio (m)",
            "nivel_m_promedio_horario.svg",
        )

    def _grafica_boxplot_nivel(self, df: pd.DataFrame) -> None:
        ancho = 1200
        alto = 700
        margen_izq = 90
        margen_sup = 70
        ancho_plot = 980
        alto_plot = 500
        resumenes = []

        for estacion, grupo in df.groupby("estacion"):
            serie = grupo[VARIABLE_OBJETIVO].dropna()
            if serie.empty:
                continue
            q1 = serie.quantile(0.25)
            q2 = serie.quantile(0.50)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            minimo = max(serie.min(), q1 - 1.5 * iqr)
            maximo = min(serie.max(), q3 + 1.5 * iqr)
            resumenes.append((estacion, minimo, q1, q2, q3, maximo))

        if not resumenes:
            self._guardar_svg("boxplot_nivel_por_estacion.svg", self._texto(600, 350, "Sin datos de nivel_m", 20))
            return

        min_y = min(r[1] for r in resumenes)
        max_y = max(r[5] for r in resumenes)
        partes = [
            self._texto(600, 35, "Distribucion de nivel_m por estacion", 22, peso="bold"),
            self._ejes(margen_izq, margen_sup, ancho_plot, alto_plot),
        ]
        paso = ancho_plot / len(resumenes)

        for idx, (estacion, minimo, q1, q2, q3, maximo) in enumerate(resumenes):
            x = margen_izq + paso * idx + paso / 2
            y_min = self._escala(minimo, min_y, max_y, margen_sup + alto_plot, margen_sup)
            y_q1 = self._escala(q1, min_y, max_y, margen_sup + alto_plot, margen_sup)
            y_q2 = self._escala(q2, min_y, max_y, margen_sup + alto_plot, margen_sup)
            y_q3 = self._escala(q3, min_y, max_y, margen_sup + alto_plot, margen_sup)
            y_max = self._escala(maximo, min_y, max_y, margen_sup + alto_plot, margen_sup)
            color = COLORES[idx % len(COLORES)]

            partes.append(f'<line x1="{x}" y1="{y_min}" x2="{x}" y2="{y_max}" stroke="{color}" stroke-width="2"/>')
            partes.append(f'<rect x="{x - 25}" y="{y_q3}" width="50" height="{y_q1 - y_q3}" fill="{color}" opacity="0.35" stroke="{color}"/>')
            partes.append(f'<line x1="{x - 25}" y1="{y_q2}" x2="{x + 25}" y2="{y_q2}" stroke="{color}" stroke-width="2"/>')
            partes.append(self._texto(x, 610, estacion, 11, color="#374151"))

        self._guardar_svg("boxplot_nivel_por_estacion.svg", "\n".join(partes), ancho, alto)

    def _grafica_correlacion(self, correlacion: pd.DataFrame) -> None:
        ancho = 1000
        alto = 950
        margen_izq = 260
        margen_sup = 90
        celda = 55
        partes = [self._texto(520, 35, "Mapa de correlacion de variables", 22, peso="bold")]

        for i, fila in enumerate(correlacion.index):
            partes.append(self._texto(250, margen_sup + i * celda + 35, fila, 10, ancla="end"))
            for j, columna in enumerate(correlacion.columns):
                valor = correlacion.loc[fila, columna]
                if pd.isna(valor):
                    color = "#f3f4f6"
                    texto = ""
                elif valor >= 0:
                    intensidad = int(255 - min(abs(valor), 1) * 130)
                    color = f"rgb({intensidad},{intensidad},255)"
                    texto = f"{valor:.2f}"
                else:
                    intensidad = int(255 - min(abs(valor), 1) * 130)
                    color = f"rgb(255,{intensidad},{intensidad})"
                    texto = f"{valor:.2f}"
                x = margen_izq + j * celda
                y = margen_sup + i * celda
                partes.append(f'<rect x="{x}" y="{y}" width="{celda}" height="{celda}" fill="{color}" stroke="#ffffff"/>')
                partes.append(self._texto(x + celda / 2, y + 33, texto, 9))

        for j, columna in enumerate(correlacion.columns):
            x = margen_izq + j * celda + celda / 2
            partes.append(
                f'<text x="{x}" y="80" font-family="Arial" font-size="10" text-anchor="start" '
                f'transform="rotate(-45 {x} 80)" fill="#111827">{html.escape(columna)}</text>'
            )

        self._guardar_svg("correlacion_variables.svg", "\n".join(partes), ancho, alto)

    def _grafica_lluvia_vs_nivel(self, df: pd.DataFrame) -> None:
        if "lluvia_mm" not in df.columns or VARIABLE_OBJETIVO not in df.columns:
            return

        diario = (
            df.set_index("fecha_hora")
            .groupby("estacion")
            .resample("D")
            .agg({
                VARIABLE_OBJETIVO: "mean",
                "lluvia_mm": "sum",
            })
            .reset_index()
        )

        diario = diario.dropna(subset=[VARIABLE_OBJETIVO, "lluvia_mm"])

        if len(diario) > 15000:
            diario = diario.sample(15000, random_state=42)

        ancho = 1000
        alto = 700
        margen_izq = 90
        margen_sup = 70
        ancho_plot = 760
        alto_plot = 500
        min_x, max_x = diario["lluvia_mm"].min(), diario["lluvia_mm"].max()
        min_y, max_y = diario[VARIABLE_OBJETIVO].min(), diario[VARIABLE_OBJETIVO].max()
        partes = [
            self._texto(500, 35, "Lluvia diaria acumulada vs nivel_m promedio diario", 20, peso="bold"),
            self._ejes(margen_izq, margen_sup, ancho_plot, alto_plot),
            self._texto(470, 650, "Lluvia diaria (mm)", 14),
            self._texto(25, 330, "Nivel promedio (m)", 14),
        ]

        for idx, (estacion, datos) in enumerate(diario.groupby("estacion")):
            color = COLORES[idx % len(COLORES)]
            for _, fila in datos.iterrows():
                x = self._escala(fila["lluvia_mm"], min_x, max_x, margen_izq, margen_izq + ancho_plot)
                y = self._escala(fila[VARIABLE_OBJETIVO], min_y, max_y, margen_sup + alto_plot, margen_sup)
                partes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}" opacity="0.35"/>')
            leyenda_y = 90 + (idx * 22)
            partes.append(f'<rect x="870" y="{leyenda_y - 10}" width="12" height="12" fill="{color}"/>')
            partes.append(self._texto(890, leyenda_y, estacion, 12, ancla="start"))

        self._guardar_svg("lluvia_vs_nivel.svg", "\n".join(partes), ancho, alto)

    def _grafica_calidad_imputacion(self, df: pd.DataFrame) -> None:
        filas = []

        for variable in VARIABLES_MODELO:
            imputado_col = f"{variable}_fue_imputado"
            if imputado_col not in df.columns:
                continue

            total = len(df)
            imputados = df[imputado_col].astype("string").str.lower().eq("true").sum()
            filas.append({
                "variable": variable,
                "imputados_pct": (imputados / total) * 100 if total else 0,
            })

        if not filas:
            return

        resumen = pd.DataFrame(filas).sort_values("imputados_pct", ascending=False)

        ancho = 1200
        alto = 700
        margen_izq = 90
        margen_sup = 70
        ancho_plot = 950
        alto_plot = 500
        max_y = max(resumen["imputados_pct"].max(), 1)
        paso = ancho_plot / len(resumen)
        partes = [
            self._texto(600, 35, "Porcentaje de imputacion por variable", 22, peso="bold"),
            self._ejes(margen_izq, margen_sup, ancho_plot, alto_plot),
            self._texto(560, 660, "Variable", 14),
            self._texto(25, 330, "Imputacion (%)", 14),
        ]

        for idx, fila in resumen.reset_index(drop=True).iterrows():
            x = margen_izq + idx * paso + paso * 0.15
            y = self._escala(fila["imputados_pct"], 0, max_y, margen_sup + alto_plot, margen_sup)
            alto_barra = margen_sup + alto_plot - y
            color = COLORES[idx % len(COLORES)]
            partes.append(f'<rect x="{x}" y="{y}" width="{paso * 0.7}" height="{alto_barra}" fill="{color}"/>')
            partes.append(self._texto(x + paso * 0.35, y - 6, f'{fila["imputados_pct"]:.1f}%', 10))
            partes.append(self._texto(x + paso * 0.35, 610, fila["variable"], 10))

        self._guardar_svg("calidad_imputacion_por_variable.svg", "\n".join(partes), ancho, alto)

    def ejecutar(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graficas_dir.mkdir(parents=True, exist_ok=True)

        df = self._cargar_datos()
        logger.info(f"Registros cargados: {len(df)}")

        self._guardar_resumen_general(df)
        self._guardar_resumen_estaciones(df)
        self._guardar_resumen_variables(df)
        self._guardar_eventos_extremos(df)
        correlacion = self._guardar_correlacion(df)

        self._grafica_nivel_por_estacion(df)
        self._grafica_promedio_mensual(df)
        self._grafica_promedio_horario(df)
        self._grafica_boxplot_nivel(df)
        self._grafica_correlacion(correlacion)
        self._grafica_lluvia_vs_nivel(df)
        self._grafica_calidad_imputacion(df)

        logger.info(f"Reportes generados en: {self.output_dir}")
        logger.info(f"Graficas generadas en: {self.graficas_dir}")

        return True


if __name__ == "__main__":
    eda = EDASeriesTiempoCAM()
    exito = eda.ejecutar()
    exit(0 if exito else 1)
