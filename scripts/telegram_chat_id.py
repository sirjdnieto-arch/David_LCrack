"""Ejecutar localmente tras enviar /start al bot; no envía mensajes."""
import getpass
import json
import urllib.request

def main():
    token=getpass.getpass('Token de BotFather (no se muestra): ').strip()
    try:
        with urllib.request.urlopen(f'https://api.telegram.org/bot{token}/getUpdates',timeout=20) as r:
            result=json.load(r)
        if not result.get('ok'): raise ValueError()
        chats={}
        for update in result.get('result',[]):
            msg=update.get('message',{})
            chat=msg.get('chat',{})
            if chat.get('type')=='private':chats[chat['id']]=chat.get('first_name','Chat privado')
        if not chats: print('No hay mensajes pendientes. Envía /start a tu bot y repite.')
        for key,name in chats.items():print(f'{name}: TELEGRAM_CHAT_ID = {key}')
    except Exception:
        print('No se pudo consultar Telegram. Comprueba el token y la conexión.')
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
