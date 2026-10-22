# David Indicators Confluence V2

App Streamlit con escáner estructural, seguimiento de siete indicadores y gráficos.

## Subir al repositorio nuevo

1. Descomprime el paquete. Abre la carpeta `david-indicators-v2`.
2. En el repositorio nuevo, usa **Add file → Upload files**. Sube el contenido de esa carpeta, de forma que `app.py` esté directamente en la raíz.
3. Comprueba que también existen `scripts/daily_snapshot.py` y `.github/workflows/daily_signals.yml` (Windows puede ocultar carpetas que empiezan por punto).
4. Si no se sube `.github`, crea el archivo desde **Add file → Create new file** e introduce como nombre completo `.github/workflows/daily_signals.yml`. Copia el contenido del archivo homónimo del ZIP.
5. En Streamlit Community Cloud crea otra app apuntando al repositorio nuevo, rama `main`, archivo `app.py`. Selecciona Python 3.12.
6. Conserva la app anterior mientras revisas esta versión.

Para ejecutar en tu ordenador:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard

- Botón Ejecutar escáner: usa el universo de `config.py` o los tickers de la barra lateral.
- Tres tipos confirmados: naranja David V2.1, ruptura alcista de cuña V3, azul RSI65/8.
- Fantasma naranja separado; no suma como confirmado.
- Orden por menor antigüedad de señal confirmada; a igualdad, más coincidencias. Fantasmas solos al final.
- Ventana predeterminada: antigüedad 0–25 velas inclusive, igual al escáner original.
- Cambio porcentual desde el cierre de la señal confirmada más reciente al último cierre descargado.
- Seguimiento original de siete indicadores y fases, con corrección del cálculo de antigüedad del cruce.
- Las divergencias informativas usan el criterio nuevo; no aumentan el recuento de siete.
- Capa BOOM y alineación 1D/1S/4H eliminadas.
- Los datos interactivos pueden incluir la vela en curso. Telegram usa exclusivamente días anteriores al día UTC actual.
- La app muestra errores de descarga y el estado de la última revisión automática cuando existe.

## Gráficos

- Nueve paneles, estética conservada. RSI con doble altura y RSI14/RSI65.
- Niveles 30/45/60/70, pendiente de la media RSI65/8 mostrada como diferencia de dos velas.
- Naranja en la vela de confirmación (pivote +3), azul en su cruce y fantasma hueco.
- Cuñas con líneas finas y discontinuas, sin relleno. Máximo seis figuras recientes visibles para evitar saturación; se conservan todos los triángulos de ruptura dentro del zoom.
- Las líneas muestran la geometría final de cada figura; no representan un replay de todas sus actualizaciones históricas.
- Se conserva la selección original de una figura principal por activo y día del motor V3, y luego su filtro por cuña.
- El nivel 60 es visual: el azul mantiene el cruce de 45 y pendiente >=0,35.

## Telegram

El workflow se ejecuta diariamente a las **00:30 UTC** (01:30 Madrid en invierno / 02:30 en verano). Recoge días anteriores, después de los cierres de los mercados americanos/europeos y de la vela diaria UTC de cripto. GitHub puede retrasar la ejecución. No se envía nada cuando no hay novedades.

1. En Telegram abre el bot oficial [@BotFather](https://t.me/BotFather), escribe `/newbot` y sigue sus pasos. Guarda el token.
2. Abre tu bot nuevo y envíale `/start`.
3. Para obtener el chat ID sin poner el token en una URL del navegador, ejecuta localmente `python scripts/telegram_chat_id.py`. Te pedirá el token de forma oculta y mostrará los chats privados que han escrito al bot. Escoge tu chat.
4. En el **repositorio nuevo**, entra en **Settings → Secrets and variables → Actions → New repository secret** y guarda:
   - `TELEGRAM_BOT_TOKEN`: token de BotFather.
   - `TELEGRAM_CHAT_ID`: identificador de tu chat.
5. En **Actions → Señales diarias Telegram → Run workflow**, ejecuta una vez. Si termina correctamente, habrá creado `snapshots/telegram_state.json` sin enviar el histórico.
6. A partir de ahí se enviarán únicamente nuevas naranjas confirmadas y cuñas rotas. Azules y fantasmas quedan en la app. Cada activo añadido se inicializa silenciosamente en su primera descarga correcta.

No incluyas token ni chat ID en archivos del repositorio. El registro persistente contiene señales, fechas y tickers; no contiene credenciales ni el destinatario.

El workflow necesita permiso de escritura para conservar `snapshots/`. Si las reglas del repositorio bloquean su commit, verás un fallo en Actions: resuélvelo antes de reintentar envíos. Usa rama `main` o cambia sus dos referencias en el YAML.

## Registro y recuperación

- Clave de aviso: ticker + tipo + fecha de confirmación.
- Primera descarga correcta por activo: baseline silenciosa; no avisa del pasado.
- Descarga fallida: no avanza el registro de ese activo; recupera señales pendientes posteriores a su baseline en la siguiente descarga correcta.
- Solo marca enviadas las señales de cada mensaje confirmado por Telegram.
- Sin señales: no hay mensaje. Fallos: se muestran en Actions y en `snapshots/status.json`, sin enviar mensajes extra por Telegram.
- Si hay muchas novedades, el resumen se divide en mensajes para respetar el límite de Telegram.
- No borres `telegram_state.json`: perderías la memoria de avisos y crearías una nueva baseline.
- Límite operativo: una caída entre la aceptación del mensaje por Telegram y la persistencia del registro, o un fallo del `git push`, puede causar un duplicado al reintentar. No se promete entrega exactamente una vez. Actions guarda una copia del registro como artefacto para recuperarla.
- Recuperación limitada al histórico descargado (5 años). Los cambios retrospectivos del proveedor pueden modificar señales; se conserva el registro ya enviado.
- Las etiquetas diarias de Yahoo se filtran por fecha de sesión; ampliar el universo a otros husos/mercados requiere revisar su cierre.

Preview local sin enviar ni modificar el registro:

```bash
python scripts/daily_snapshot.py --tickers AAPL BTC-USD
```

En un registro nuevo esta preview mostrará cero novedades por la inicialización silenciosa.

Pruebas del sistema de avisos (sin Telegram real):

```bash
python -m unittest discover -s tests -v
```

Documentación oficial: [Telegram Bot API](https://core.telegram.org/bots/api#sendmessage), [creación del bot](https://core.telegram.org/bots/tutorial#obtain-your-bot-token), [workflows de GitHub](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
