import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('alerts',ROOT/'scripts/daily_snapshot.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def events(date,kind='DIV_ALCISTA'):
    return pd.DataFrame([dict(Senal=kind,Estado='CONFIRMADA',Fecha=pd.Timestamp(date),
        Velas_hace=0,Precio_cierre=100,Cambio_desde_senal_pct=1)])

class AlertsTests(unittest.TestCase):
    def test_baseline_new_repeat_and_recovery(self):
        state={'version':1,'tickers':{}}
        self.assertEqual(a.new_events(state,'TEST',events('2026-01-01'),'2026-01-01'),[])
        pending=a.new_events(state,'TEST',events('2026-01-03'),'2026-01-05')
        self.assertEqual(len(pending),1)  # missed run is recovered
        self.assertEqual(len(a.new_events(state,'TEST',events('2026-01-03'),'2026-01-06')),1)
        state['tickers']['TEST']['seen'].append(pending[0]['id'])
        self.assertEqual(a.new_events(state,'TEST',events('2026-01-03'),'2026-01-06'),[])
        self.assertEqual(a.new_events(state,'TEST',events('2026-01-07','RSI_AZUL'),'2026-01-07'),[])
        self.assertEqual(a.new_events(state,'TEST',events('2026-01-07','DIV_FANTASMA'),'2026-01-07'),[])
    def test_closed_daily_and_silence(self):
        d=pd.DataFrame({'Close':[1,2]},index=pd.to_datetime(['2026-01-01','2026-01-02'],utc=True))
        self.assertEqual(len(a.closed_daily(d,datetime(2026,1,2,0,30,tzinfo=timezone.utc))),1)
        self.assertEqual(list(a.message_batches([])),[])
    def test_delivery_failure_does_not_acknowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state_path=root/'snapshots/telegram_state.json'
            state={'version':1,'tickers':{'TEST':{'baseline':'2020-01-01','seen':[]}}}
            a.save_json(state_path,state)
            d=pd.DataFrame({'Close':[100]*300},index=pd.bdate_range('2025-01-01',periods=300))
            with patch.object(a,'ROOT',root),patch.object(a,'download_df',return_value=d),\
                 patch.object(a,'structural_data',return_value={'events':events('2025-02-01')}),\
                 patch.dict(a.os.environ,{'TELEGRAM_BOT_TOKEN':'test','TELEGRAM_CHAT_ID':'test'}),\
                 patch.object(a,'send_telegram',side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError):a.run(['TEST'],state_path,True)
                self.assertEqual(a.load_state(state_path)['tickers']['TEST']['seen'],[])
                with patch.object(a,'send_telegram') as send:
                    self.assertEqual(a.run(['TEST'],state_path,True),0)
                    send.assert_called_once()
                    self.assertEqual(a.run(['TEST'],state_path,True),0)
                    send.assert_called_once()
    def test_message_chunking(self):
        rows=[dict(events('2026-01-01').iloc[0],Ticker=f'ASSET{i}',id=str(i)) for i in range(100)]
        batches=list(a.message_batches(rows))
        self.assertEqual(sum(len(m) for _,m in batches),100)
        self.assertTrue(all(len(text)<=3500 for text,_ in batches))
if __name__=='__main__':unittest.main()
