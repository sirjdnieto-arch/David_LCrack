"""Escaneo diario, registro persistente y resumen Telegram solo con novedades."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
import pandas as pd
from config import ALL_TICKERS, TICKER_NAMES
from indicators import download_df
from structural_signals import structural_data, LABELS

ALERT_TYPES = {'DIV_ALCISTA', 'CUNA_ROTA'}
STATE_PATH = ROOT / 'snapshots/telegram_state.json'

def closed_daily(df, now=None):
    """Conservador: excluye la fecha UTC actual. Ejecutar a las 00:30 UTC.

    El universo configurado contiene mercados americanos/europeos y cripto.
    Conserva la fecha de sesión de Yahoo, sin convertir etiquetas a otra fecha.
    """
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    return df.loc[[ts.date() < today for ts in df.index]].copy()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def load_state(path):
    if not path.exists(): return {'version': 1, 'tickers': {}}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('tickers'), dict):
        raise ValueError('Registro de avisos inválido; no se reinicia automáticamente')
    return data

def event_id(ticker, event):
    return f"{ticker}|{event['Senal']}|{pd.Timestamp(event['Fecha']):%Y-%m-%d}"

def new_events(state, ticker, events, last_date):
    """Primera lectura de cada ticker: baseline silenciosa, sin avisos históricos."""
    events = events[events.Senal.isin(ALERT_TYPES) & (events.Estado == 'CONFIRMADA')]
    records = events.to_dict('records')
    if ticker not in state['tickers']:
        state['tickers'][ticker] = {
            'baseline': str(pd.Timestamp(last_date).date()),
            'seen': [event_id(ticker, e) for e in records],
        }
        return []
    info = state['tickers'][ticker]
    seen = set(info['seen'])
    return [dict(e, Ticker=ticker, id=event_id(ticker,e)) for e in records
            if event_id(ticker,e) not in seen
            and pd.Timestamp(e['Fecha']).date() > pd.Timestamp(info['baseline']).date()]

def message_batches(events, limit=3500):
    header = 'David · nuevas señales confirmadas (1D)\n'
    text = header; members = []
    for e in sorted(events, key=lambda e: (int(e['Velas_hace']),e['Ticker'],e['Senal'])):
        line = (f"\n{e['Ticker']} · {TICKER_NAMES.get(e['Ticker'],e['Ticker'])}\n"
                f"{LABELS[e['Senal']]} · {pd.Timestamp(e['Fecha']):%d/%m/%Y}"
                f" · hace {int(e['Velas_hace'])} velas\n"
                f"Cierre señal: {float(e['Precio_cierre']):.2f}"
                f" · Desde señal: {float(e['Cambio_desde_senal_pct']):+.2f}%\n")
        if len(text+line)>limit and members:
            yield text, members
            text = header; members = []
        text += line; members.append(e)
    if members: yield text, members

def send_telegram(text, token, chat_id):
    payload = json.dumps({'chat_id': chat_id, 'text': text}).encode()
    req = urllib.request.Request(f'https://api.telegram.org/bot{token}/sendMessage',
                                 data=payload,headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        if not result.get('ok'): raise RuntimeError('Telegram rechazó el mensaje')
    except Exception:
        # Never log request URLs: they contain the secret bot token.
        raise RuntimeError('No se pudo confirmar el envío a Telegram; avisos pendientes') from None

def run(tickers, state_path=STATE_PATH, send=False):
    now = datetime.now(timezone.utc)
    token = os.environ.get('TELEGRAM_BOT_TOKEN','')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID','')
    if send and (not token or not chat_id):
        raise RuntimeError('Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en GitHub Secrets')
    state = load_state(state_path)
    pending = []; status = []; recent = []
    for ticker in dict.fromkeys(tickers):
        try:
            df = closed_daily(download_df(ticker,period='5y',interval='1d'),now)
            if len(df)<260: raise ValueError('Menos de 260 velas cerradas disponibles')
            events = structural_data(df,ticker)['events'].copy()
            events['Cambio_desde_senal_pct'] = (float(df.Close.iloc[-1])/events.Precio_cierre-1)*100
            ev = events.loc[events.Velas_hace<=25].copy(); ev['Ticker']=ticker
            recent.extend(ev.assign(Fecha=ev.Fecha.astype(str)).to_dict('records'))
            pending.extend(new_events(state,ticker,events,df.index[-1]))
            status.append({'Ticker':ticker,'Estado':'OK','Ultima_vela':str(df.index[-1].date())})
        except Exception as exc:
            status.append({'Ticker':ticker,'Estado':'ERROR','Error':type(exc).__name__})
        print(f"{ticker}: {status[-1]['Estado']}",flush=True)
    # --preview does not initialize or mutate the persistent ledger.
    if not send:
        for text,_ in message_batches(pending): print(text)
        print(f'PREVIEW: {len(pending)} novedades; no se envía ni se guarda el registro.')
        return 1 if any(s['Estado']=='ERROR' for s in status) else 0
    save_json(state_path,state)  # persist baselines even if delivery later fails
    save_json(ROOT/'snapshots/status.json',{'checked_at':now.isoformat(),'assets':status})
    pd.DataFrame(recent).to_csv(ROOT/'snapshots/latest_signals.csv',index=False)
    for text,members in message_batches(pending):
        send_telegram(text,token,chat_id)
        for e in members:
            state['tickers'][e['Ticker']]['seen'].append(e['id'])
        save_json(state_path,state)  # only confirmed successful batches are acknowledged
    print(f'{len(pending)} novedades enviadas.' if pending else 'Sin novedades: no se envía Telegram.')
    return 1 if any(s['Estado']=='ERROR' for s in status) else 0

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--send',action='store_true',help='Enviar y actualizar registro; por defecto solo preview')
    parser.add_argument('--tickers',nargs='+',default=ALL_TICKERS)
    args=parser.parse_args()
    try: sys.exit(run(args.tickers,send=args.send))
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
