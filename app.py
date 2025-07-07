# app.py (Backend con Flask, Lógica de Negocio y Simulación de Gemini AI)
# VERSIÓN 11.0 - LISTO PARA RENDER

# --- Imports Nativos y de Flask ---
import io
import math
import traceback
import json
import time
import random
from flask import Flask, jsonify, request, Response

# --- Imports de Librerías de Terceros ---
import pandas as pd
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from flask_cors import CORS

# ==============================================================================
# --- CONFIGURACIÓN INICIAL Y ALMACENAMIENTO EN MEMORIA ---
# ==============================================================================
# Al usar static_folder='.', le decimos a Flask que sirva los archivos desde la raíz
# donde también estará index.html
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

paradas_en_memoria = {}
depot_en_memoria = None
DEPOT_ID = "depot"
COSTO_POR_KM_COP = 532

# ==============================================================================
# --- SIMULACIÓN DE LA API DE GEMINI ---
# ==============================================================================
def analizar_plan_con_ia(datos_optimizacion):
    print("🤖 Iniciando análisis logístico detallado con IA (Simulación de Gemini)...")
    time.sleep(random.uniform(1, 2))
    rutas_locales = datos_optimizacion.get('rutas_locales', [])
    num_rutas = len(rutas_locales)
    if not num_rutas:
        return {"titulo": "Análisis Logístico por Rout-IA", "insights": ["No hay rutas para analizar."]}
    
    total_pasajeros = sum(r.get('total_pasajeros', 0) for r in rutas_locales)
    distancia_total_km = sum(r.get('distancia_optima_m', 0) for r in rutas_locales) / 1000
    costo_total_operacion = sum(r.get('costo_estimado_cop', 0) for r in rutas_locales)
    
    utilizaciones = [r.get('capacidad_utilizada_pct', 0) for r in rutas_locales]
    utilizacion_promedio = sum(utilizaciones) / len(utilizaciones) if utilizaciones else 0
    
    insights = [
        f"Resumen General: Se planificaron {num_rutas} rutas para cubrir {total_pasajeros} pasajeros.",
        f"Costo Operativo Total: El costo estimado para toda la operación es de ${costo_total_operacion:,.0f} COP.",
        f"Utilización Promedio de Flota: {utilizacion_promedio:.1f}%.",
    ]

    if num_rutas > 1:
        ruta_mas_costosa = max(rutas_locales, key=lambda r: r['costo_estimado_cop'])
        insights.append(
            f"Ruta de Mayor Costo: La ruta del Vehículo {ruta_mas_costosa['vehiculo_id']} es la más cara (${ruta_mas_costosa['costo_estimado_cop']:,.0f} COP)."
        )

    if utilizacion_promedio < 60:
        insights.append("Sugerencia de Eficiencia: La flota parece subutilizada. Evalúe usar vehículos de menor capacidad para reducir costos.")
    elif utilizacion_promedio > 95:
        insights.append("Alerta de Capacidad: Varios vehículos operan cerca de su límite. Esto reduce la flexibilidad ante imprevistos.")
    
    return {
        "id_analisis": f"gemini-analysis-{int(time.time())}",
        "titulo": "Análisis Logístico por Rout-IA",
        "insights": insights,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# ==============================================================================
# --- LÓGICA DE NEGOCIO Y FUNCIONES AUXILIARES ---
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    if not all(isinstance(x, (float, int)) for x in [lat1, lon1, lat2, lon2]): return 0.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    return 6371000 * 2 * math.asin(min(1.0, math.sqrt(a)))

def _parse_csv(file_stream):
    try:
        df = pd.read_csv(file_stream, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        lat_col = next((c for c in df.columns if c in ['lat', 'latitude', 'latitud']), None)
        lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude', 'longitud']), None)
        if not lat_col or not lon_col: raise ValueError("Columnas de lat/lon no encontradas.")
        dem_col = next((c for c in df.columns if c in ['pasajeros', 'demanda', 'demand']), None)
        nom_col = next((c for c in df.columns if c in ['nombre', 'name']), None)
        paradas = []
        for index, row in df.iterrows():
            try:
                parada_id = f"parada_{int(time.time())}_{index}"
                paradas.append({
                    'id': parada_id, 'lat': float(row[lat_col]), 'lon': float(row[lon_col]),
                    'pasajeros': int(row[dem_col]) if dem_col and pd.notna(row[dem_col]) else 1,
                    'nombre': str(row[nom_col]) if nom_col and pd.notna(row[nom_col]) else f'Parada {index + 1}'
                })
            except (ValueError, TypeError): continue
        return paradas
    except Exception as e:
        raise ValueError(f"Error procesando CSV: {e}")

# ==============================================================================
# --- RUTAS DE LA API (ENDPOINTS) ---
# ==============================================================================
@app.route('/')
def serve_index(): return app.send_static_file('index.html')

@app.route('/api/paradas', methods=['GET'])
def get_paradas(): return jsonify(list(paradas_en_memoria.values())), 200

@app.route('/api/parada', methods=['POST'])
def add_parada():
    data = request.get_json()
    if not data or 'lat' not in data or 'lon' not in data:
        return jsonify({"error": "Datos incompletos para crear la parada."}), 400
    
    parada_id = f"parada_{int(time.time() * 1000)}"
    nueva_parada = {
        'id': parada_id,
        'lat': data['lat'],
        'lon': data['lon'],
        'pasajeros': int(data.get('pasajeros', 1)),
        'nombre': data.get('nombre', f'Parada {parada_id[-4:]}')
    }
    paradas_en_memoria[parada_id] = nueva_parada
    print(f"➕ Parada agregada manualmente: {parada_id}")
    return jsonify(nueva_parada), 201

@app.route('/api/parada/<parada_id>', methods=['DELETE'])
def delete_parada(parada_id):
    if parada_id in paradas_en_memoria:
        del paradas_en_memoria[parada_id]
        print(f"➖ Parada eliminada: {parada_id}")
        return jsonify({"message": "Parada eliminada con éxito"}), 200
    else:
        return jsonify({"error": "Parada no encontrada"}), 404

@app.route('/api/upload_paradas', methods=['POST'])
def upload_paradas_file():
    if 'paradasFile' not in request.files: return jsonify({"error": "No se encontró archivo"}), 400
    file = request.files['paradasFile']
    if not file or not file.filename.endswith('.csv'): return jsonify({"error": "Archivo no permitido, solo .csv"}), 400
    try:
        paradas_cargadas = _parse_csv(io.TextIOWrapper(file.stream, encoding='utf-8'))
        paradas_en_memoria.clear()
        for p_data in paradas_cargadas: paradas_en_memoria[p_data['id']] = p_data
        print(f"📄 Se cargaron {len(paradas_cargadas)} paradas.")
        return jsonify({"message": f"{len(paradas_cargadas)} paradas cargadas."}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al procesar el archivo: {e}"}), 400

@app.route('/api/clear', methods=['POST'])
def clear_data():
    paradas_en_memoria.clear()
    global depot_en_memoria
    depot_en_memoria = None
    print("🗑️ Datos en memoria eliminados.")
    return jsonify({"message": "Todos los datos han sido eliminados."}), 200

@app.route('/api/optimize', methods=['POST'])
def optimize_routes_full():
    global depot_en_memoria
    data = request.get_json()
    if not data or 'vehiculos' not in data or 'depot' not in data:
        return jsonify({"error": "Datos de entrada incompletos"}), 400
    
    depot_en_memoria = data['depot']
    if not paradas_en_memoria:
        return jsonify({"error": "No hay paradas para optimizar"}), 400

    try:
        paradas_df = pd.DataFrame(list(paradas_en_memoria.values()))
        vehiculos_df = pd.DataFrame(data['vehiculos'])
        vehiculos_df['ID_Vehiculo'] = vehiculos_df.index + 1
        
        if paradas_df['pasajeros'].sum() > vehiculos_df['capacidad'].sum():
            return jsonify({"error": "Capacidad total de la flota es insuficiente para la demanda."}), 400

        print("🧠 Ejecutando algoritmo de optimización por capacidad (Cero Sobrecupo)...")
        
        paradas_pendientes_df = paradas_df.copy()
        rutas_asignadas = {v_id: [] for v_id in vehiculos_df['ID_Vehiculo']}
        capacidad_restante = {row['ID_Vehiculo']: row['capacidad'] for _, row in vehiculos_df.iterrows()}

        for vehiculo_id in vehiculos_df['ID_Vehiculo']:
            if paradas_pendientes_df.empty: break
            
            while True:
                paradas_que_caben = paradas_pendientes_df[paradas_pendientes_df['pasajeros'] <= capacidad_restante[vehiculo_id]].copy()
                if paradas_que_caben.empty:
                    break

                if not rutas_asignadas[vehiculo_id]:
                    last_lat, last_lon = depot_en_memoria['lat'], depot_en_memoria['lon']
                else:
                    last_parada_id = rutas_asignadas[vehiculo_id][-1]
                    last_parada_info = paradas_df[paradas_df['id'] == last_parada_id].iloc[0]
                    last_lat, last_lon = last_parada_info['lat'], last_parada_info['lon']

                paradas_que_caben.loc[:, 'dist'] = paradas_que_caben.apply(
                    lambda row: haversine(last_lat, last_lon, row['lat'], row['lon']), axis=1
                )
                mejor_parada = paradas_que_caben.sort_values('dist').iloc[0]
                
                rutas_asignadas[vehiculo_id].append(mejor_parada['id'])
                capacidad_restante[vehiculo_id] -= mejor_parada['pasajeros']
                paradas_pendientes_df = paradas_pendientes_df.drop(mejor_parada.name)

        paradas_df_indexed = paradas_df.set_index('id')
        depot_df = pd.DataFrame([{'lat': depot_en_memoria['lat'], 'lon': depot_en_memoria['lon'], 'pasajeros': 0}], index=[DEPOT_ID])
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

            distancia_total_m = sum(
                haversine(
                    paradas_con_depot_df.loc[secuencia_final_ids[i]]['lat'], 
                    paradas_con_depot_df.loc[secuencia_final_ids[i]]['lon'], 
                    paradas_con_depot_df.loc[secuencia_final_ids[i+1]]['lat'], 
                    paradas_con_depot_df.loc[secuencia_final_ids[i+1]]['lon']
                ) for i in range(len(secuencia_final_ids) - 1)
            )
            
            distancia_total_km = distancia_total_m / 1000
            costo_estimado = distancia_total_km * COSTO_POR_KM_COP

            vehiculo_info = vehiculos_df[vehiculos_df['ID_Vehiculo'] == vehiculo_id].iloc[0]
            
            ids_paradas_ordenadas = [pid for pid in secuencia_final_ids if pid != DEPOT_ID]
            paradas_info_ordenadas = paradas_df_indexed.loc[ids_paradas_ordenadas].reset_index().to_dict('records')
            
            ruta_resultado = {
                "vehiculo_id": int(vehiculo_id),
                "secuencia_paradas": ids_paradas_ordenadas,
                "paradas_info": paradas_info_ordenadas,
                "total_pasajeros": int(paradas_df_indexed.loc[paradas_ids].sum()['pasajeros']),
                "capacidad": int(vehiculo_info['capacidad']),
                "capacidad_utilizada_pct": (int(paradas_df_indexed.loc[paradas_ids].sum()['pasajeros']) / int(vehiculo_info['capacidad'])) * 100,
                "distancia_optima_m": distancia_total_m,
                "costo_estimado_cop": costo_estimado
            }
            resultados_locales.append(ruta_resultado)
        
        print("✅ Optimización local completada.")
        return jsonify({"rutas_locales": resultados_locales, "depot": depot_en_memoria, "paradas_utilizadas": list(paradas_en_memoria.values())})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error interno en optimización: {str(e)}"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_with_ai():
    optimization_results = request.get_json()
    if not optimization_results: return jsonify({"error": "No se proporcionaron resultados para analizar"}), 400
    ai_analysis = analizar_plan_con_ia(optimization_results)
    return jsonify(ai_analysis)

# El bloque if __name__ == '__main__': se elimina o comenta,
# ya que Gunicorn se encargará de ejecutar la aplicación.
