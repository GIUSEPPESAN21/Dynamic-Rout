# app.py (Aplicación de Optimización de Rutas con Streamlit)
# VERSIÓN 12.0 - TODO EN UNO

import streamlit as st
import pandas as pd
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
import math
import io
import time
import random

# ==============================================================================
# --- CONFIGURACIÓN DE LA PÁGINA ---
# ==============================================================================
st.set_page_config(
    page_title="Rout Now | Logística Inteligente",
    page_icon="🚚",
    layout="wide"
)

# ==============================================================================
# --- LÓGICA DE NEGOCIO Y FUNCIONES AUXILIARES ---
# ==============================================================================
# Constantes
DEPOT_ID = "depot"
COSTO_POR_KM_COP = 532

def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine entre dos puntos en metros."""
    if not all(isinstance(x, (float, int)) for x in [lat1, lon1, lat2, lon2]): return 0.0
    R = 6371000  # Radio de la Tierra en metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _parse_csv(file_stream):
    """Parsea un archivo CSV de paradas."""
    try:
        df = pd.read_csv(file_stream, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        
        # Mapeo flexible de columnas
        col_map = {
            'lat': ['lat', 'latitude', 'latitud'],
            'lon': ['lon', 'lng', 'longitude', 'longitud'],
            'pasajeros': ['pasajeros', 'demanda', 'demand'],
            'nombre': ['nombre', 'name']
        }
        
        parsed_cols = {}
        for key, possible_names in col_map.items():
            found_col = next((c for c in df.columns if c in possible_names), None)
            if not found_col and key in ['lat', 'lon']:
                raise ValueError(f"No se encontraron columnas de coordenadas ({key}).")
            parsed_cols[key] = found_col

        # Crear un nuevo DataFrame con nombres de columna estandarizados
        paradas_df = pd.DataFrame()
        paradas_df['lat'] = df[parsed_cols['lat']].astype(float)
        paradas_df['lon'] = df[parsed_cols['lon']].astype(float)
        paradas_df['pasajeros'] = df[parsed_cols['pasajeros']].astype(int) if parsed_cols['pasajeros'] else 1
        paradas_df['nombre'] = df[parsed_cols['nombre']].astype(str) if parsed_cols['nombre'] else 'Parada sin nombre'
        paradas_df['id'] = [f"parada_{int(time.time() * 1000)}_{i}" for i in range(len(paradas_df))]
        
        return paradas_df
    except Exception as e:
        st.error(f"Error procesando CSV: {e}")
        return pd.DataFrame()

def run_optimization(paradas_df, vehiculos, depot_coords):
    """Ejecuta el algoritmo de optimización de rutas."""
    if paradas_df.empty:
        st.warning("No hay paradas para optimizar.")
        return None

    vehiculos_df = pd.DataFrame(vehiculos)
    vehiculos_df['ID_Vehiculo'] = vehiculos_df.index + 1

    if paradas_df['pasajeros'].sum() > vehiculos_df['capacidad'].sum():
        st.error("Capacidad total de la flota es insuficiente para la demanda.")
        return None

    paradas_pendientes_df = paradas_df.copy()
    rutas_asignadas = {v_id: [] for v_id in vehiculos_df['ID_Vehiculo']}
    capacidad_restante = {row['ID_Vehiculo']: row['capacidad'] for _, row in vehiculos_df.iterrows()}

    for vehiculo_id in vehiculos_df['ID_Vehiculo']:
        if paradas_pendientes_df.empty: break
        while True:
            paradas_que_caben = paradas_pendientes_df[paradas_pendientes_df['pasajeros'] <= capacidad_restante[vehiculo_id]].copy()
            if paradas_que_caben.empty: break

            if not rutas_asignadas[vehiculo_id]:
                last_lat, last_lon = depot_coords['lat'], depot_coords['lon']
            else:
                last_parada_id = rutas_asignadas[vehiculo_id][-1]
                last_parada_info = paradas_df[paradas_df['id'] == last_parada_id].iloc[0]
                last_lat, last_lon = last_parada_info['lat'], last_parada_info['lon']

            paradas_que_caben.loc[:, 'dist'] = paradas_que_caben.apply(lambda row: haversine(last_lat, last_lon, row['lat'], row['lon']), axis=1)
            mejor_parada = paradas_que_caben.sort_values('dist').iloc[0]
            
            rutas_asignadas[vehiculo_id].append(mejor_parada['id'])
            capacidad_restante[vehiculo_id] -= mejor_parada['pasajeros']
            paradas_pendientes_df = paradas_pendientes_df.drop(mejor_parada.name)

    paradas_df_indexed = paradas_df.set_index('id')
    depot_df = pd.DataFrame([{'lat': depot_coords['lat'], 'lon': depot_coords['lon'], 'pasajeros': 0, 'nombre': 'Depósito'}], index=[DEPOT_ID])
    paradas_con_depot_df = pd.concat([paradas_df_indexed, depot_df])
    
    resultados_locales = []
    for vehiculo_id, paradas_ids in rutas_asignadas.items():
        if not paradas_ids: continue
        
        nodos_ids = [DEPOT_ID] + paradas_ids
        dist_matrix = np.array([[haversine(paradas_con_depot_df.loc[i]['lat'], paradas_con_depot_df.loc[i]['lon'], paradas_con_depot_df.loc[j]['lat'], paradas_con_depot_df.loc[j]['lon']) for j in nodos_ids] for i in nodos_ids])
        permutation, _ = solve_tsp_simulated_annealing(dist_matrix)
        secuencia_ids_ordenada = [nodos_ids[i] for i in permutation]
        start_idx = secuencia_ids_ordenada.index(DEPOT_ID)
        secuencia_final_ids = secuencia_ids_ordenada[start_idx:] + secuencia_ids_ordenada[:start_idx] + [DEPOT_ID]

        distancia_total_m = sum(haversine(paradas_con_depot_df.loc[secuencia_final_ids[i]]['lat'], paradas_con_depot_df.loc[secuencia_final_ids[i]]['lon'], paradas_con_depot_df.loc[secuencia_final_ids[i+1]]['lat'], paradas_con_depot_df.loc[secuencia_final_ids[i+1]]['lon']) for i in range(len(secuencia_final_ids) - 1))
        
        distancia_total_km = distancia_total_m / 1000
        costo_estimado = distancia_total_km * COSTO_POR_KM_COP

        vehiculo_info = vehiculos_df[vehiculos_df['ID_Vehiculo'] == vehiculo_id].iloc[0]
        ids_paradas_ordenadas = [pid for pid in secuencia_final_ids if pid != DEPOT_ID]
        paradas_info_ordenadas = paradas_df_indexed.loc[ids_paradas_ordenadas].reset_index().to_dict('records')
        
        resultados_locales.append({
            "vehiculo_id": int(vehiculo_id),
            "secuencia_paradas": ids_paradas_ordenadas,
            "paradas_info": paradas_info_ordenadas,
            "total_pasajeros": int(paradas_df_indexed.loc[paradas_ids].sum()['pasajeros']),
            "capacidad": int(vehiculo_info['capacidad']),
            "capacidad_utilizada_pct": (int(paradas_df_indexed.loc[paradas_ids].sum()['pasajeros']) / int(vehiculo_info['capacidad'])) * 100,
            "distancia_optima_m": distancia_total_m,
            "costo_estimado_cop": costo_estimado
        })
    
    return {"rutas_locales": resultados_locales}

def analizar_plan_con_ia(datos_optimizacion):
    """Simula un análisis de IA sobre los resultados."""
    with st.spinner("🤖 Analizando plan de rutas con IA..."):
        time.sleep(random.uniform(1, 2))
        rutas_locales = datos_optimizacion.get('rutas_locales', [])
        num_rutas = len(rutas_locales)
        if not num_rutas:
            return {"titulo": "Análisis Logístico por Rout-IA", "insights": ["No hay rutas para analizar."]}
        
        total_pasajeros = sum(r.get('total_pasajeros', 0) for r in rutas_locales)
        costo_total_operacion = sum(r.get('costo_estimado_cop', 0) for r in rutas_locales)
        utilizaciones = [r.get('capacidad_utilizada_pct', 0) for r in rutas_locales]
        utilizacion_promedio = sum(utilizaciones) / len(utilizaciones) if utilizaciones else 0
        
        insights = [
            f"**Costo Operativo Total:** El costo estimado para toda la operación es de **${costo_total_operacion:,.0f} COP**.",
            f"**Utilización Promedio de Flota:** {utilizacion_promedio:.1f}%.",
        ]

        if num_rutas > 1:
            ruta_mas_costosa = max(rutas_locales, key=lambda r: r['costo_estimado_cop'])
            insights.append(f"**Ruta de Mayor Costo:** La del Vehículo {ruta_mas_costosa['vehiculo_id']} (${ruta_mas_costosa['costo_estimado_cop']:,.0f} COP).")

        if utilizacion_promedio < 60:
            insights.append("**Sugerencia de Eficiencia:** La flota parece subutilizada. Evalúe usar vehículos de menor capacidad para reducir costos.")
        elif utilizacion_promedio > 95:
            insights.append("**Alerta de Capacidad:** Varios vehículos operan cerca de su límite, lo que reduce la flexibilidad.")
        
        return {
            "titulo": "Análisis Logístico por Rout-IA",
            "insights": insights,
        }

# ==============================================================================
# --- INICIALIZACIÓN DEL ESTADO DE LA APLICACIÓN ---
# ==============================================================================
if 'paradas_df' not in st.session_state:
    st.session_state.paradas_df = pd.DataFrame()
if 'vehiculos' not in st.session_state:
    st.session_state.vehiculos = [{"capacidad": 35}, {"capacidad": 40}]
if 'depot_coords' not in st.session_state:
    st.session_state.depot_coords = {'lat': 3.9039, 'lon': -76.2987} # Buga, Colombia
if 'rutas_calculadas' not in st.session_state:
    st.session_state.rutas_calculadas = None
if 'analisis_ia' not in st.session_state:
    st.session_state.analisis_ia = None

# ==============================================================================
# --- INTERFAZ DE USUARIO (SIDEBAR) ---
# ==============================================================================
with st.sidebar:
    st.title("🚚 Rout Now")
    st.subheader("Configuración de la Operación")

    with st.expander("1. Cargar Paradas (CSV)", expanded=True):
        uploaded_file = st.file_uploader("Sube tu archivo de paradas", type=["csv"])
        if uploaded_file is not None:
            df = _parse_csv(uploaded_file)
            if not df.empty:
                st.session_state.paradas_df = df
                st.success(f"Se cargaron {len(df)} paradas.")
                st.session_state.rutas_calculadas = None # Resetear resultados
                st.session_state.analisis_ia = None

    with st.expander("2. Flota de Vehículos", expanded=True):
        for i, vehiculo in enumerate(st.session_state.vehiculos):
            cols = st.columns([3, 1])
            new_cap = cols[0].number_input(f"Capacidad Vehículo {i+1}", min_value=1, value=vehiculo['capacidad'], key=f"v_cap_{i}")
            st.session_state.vehiculos[i]['capacidad'] = new_cap
            if cols[1].button("➖", key=f"del_v_{i}"):
                st.session_state.vehiculos.pop(i)
                st.rerun()
        if st.button("Añadir Vehículo", use_container_width=True):
            st.session_state.vehiculos.append({"capacidad": 20})
            st.rerun()

    with st.expander("3. Ubicación del Depósito", expanded=True):
        st.session_state.depot_coords['lat'] = st.number_input("Latitud del Depósito", value=st.session_state.depot_coords['lat'], format="%.6f")
        st.session_state.depot_coords['lon'] = st.number_input("Longitud del Depósito", value=st.session_state.depot_coords['lon'], format="%.6f")

    st.header("Acciones")
    if st.button("🚀 Optimizar Rutas", type="primary", use_container_width=True):
        with st.spinner("Calculando las rutas más eficientes..."):
            st.session_state.rutas_calculadas = run_optimization(
                st.session_state.paradas_df,
                st.session_state.vehiculos,
                st.session_state.depot_coords
            )
            st.session_state.analisis_ia = None # Resetear análisis anterior

    if st.session_state.rutas_calculadas:
        if st.button("🤖 Analizar con IA", use_container_width=True):
            st.session_state.analisis_ia = analizar_plan_con_ia(st.session_state.rutas_calculadas)

# ==============================================================================
# --- ÁREA PRINCIPAL DE VISUALIZACIÓN ---
# ==============================================================================
st.header("Mapa de Operación")

# Preparar datos para el mapa
if not st.session_state.paradas_df.empty:
    map_data = st.session_state.paradas_df[['lat', 'lon']].copy()
    depot_df = pd.DataFrame([st.session_state.depot_coords])
    st.map(pd.concat([map_data, depot_df]), zoom=12)
else:
    st.map(pd.DataFrame([st.session_state.depot_coords]), zoom=12)
    st.info("Carga un archivo CSV o añade paradas para comenzar.")

st.header("Resultados de la Optimización")

if st.session_state.rutas_calculadas:
    rutas = st.session_state.rutas_calculadas.get("rutas_locales", [])
    if not rutas:
        st.warning("No se pudieron generar rutas con los datos proporcionados.")
    else:
        for i, ruta in enumerate(rutas):
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Vehículo", f"#{ruta['vehiculo_id']}")
                col2.metric("Total Paradas", len(ruta['secuencia_paradas']))
                col3.metric("Costo Estimado", f"${ruta['costo_estimado_cop']:,.0f}")
                
                st.progress(int(ruta['capacidad_utilizada_pct']), text=f"Utilización: {ruta['total_pasajeros']} / {ruta['capacidad']} ({ruta['capacidad_utilizada_pct']:.1f}%)")

                with st.expander("Ver secuencia de la ruta"):
                    nombres_paradas = [p['nombre'] for p in ruta['paradas_info']]
                    secuencia_str = " → ".join(nombres_paradas)
                    st.info(f"Depósito → {secuencia_str} → Depósito")
else:
    st.info("Haz clic en 'Optimizar Rutas' para ver los resultados aquí.")

if st.session_state.analisis_ia:
    st.header(st.session_state.analisis_ia['titulo'])
    for insight in st.session_state.analisis_ia['insights']:
        st.markdown(f"- {insight}")
