import json
from google.cloud import firestore
from google.oauth2 import service_account

# Leer user_id del config
with open('config/infractivision_config.json', 'r') as f:
    config = json.load(f)

user_id = config['user_id']

# Conectar a Firestore
creds = service_account.Credentials.from_service_account_file('secrets/infractivision-474103-0f907d0fbc62.json')
db = firestore.Client(project='infractivision-474103', credentials=creds)

print(f'Usuario: {user_id}')

# Verificar infracciones
infracciones = db.collection('usuarios').document(user_id).collection('infracciones').get()
print(f'Infracciones: {len(infracciones)}')

if infracciones:
    for i, doc in enumerate(infracciones[:2]):
        data = doc.to_dict()
        print(f'  {i+1}. Placa: {data.get("placa")}')
        print(f'     Fecha: {data.get("fecha")}')
        print(f'     Ubicacion: {data.get("ubicacion")}')
        print(f'     Confianza: {data.get("confianza")}')

# Verificar indicadores  
indicadores = db.collection('usuarios').document(user_id).collection('indicadores').get()
print(f'Indicadores: {len(indicadores)}')

if indicadores:
    data = indicadores[0].to_dict()
    print(f'  TI Acierto: {data.get("indicadores_TI_porcentaje_acierto")}%')
    print(f'  Fecha generacion: {data.get("fecha_generacion")}')