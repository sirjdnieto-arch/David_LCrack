"""Motores originales David V2.1 y cuñas V3; adaptación de salida para la app."""
import numpy as np
import pandas as pd
from itertools import combinations
MOSTRAR_FANTASMAS = True
PARAMS = dict(
    rsi65_len=65, rsi65_ma_len=8, nivel_rsi65=45.0,
    pendiente_minima=0.35, rsi14_len=14, atr_len=14,
    pivot_left=3, pivot_right=3,
    min_barras=5, max_barras=50,
    min_rsi_diff=3.0, min_price_atr=0.25,
    nivel_sobreventa=30.0, mostrar_fantasmas=MOSTRAR_FANTASMAS,
)
L, R, ATR_N = 5, 5, 14
MIN_DURACION, MAX_DURACION, MIN_SEP = 25, 180, 8
TOL_PIVOTE_ATR, TOL_CIERRE_ATR = .70, .50
DESCENSO_MIN_ATR, ANCHURA_MIN_ATR = .70, .65
CANAL_MIN_CONTRACCION, CANAL_MAX_CONTRACCION, CUNA_MAX_CONTRACCION = -.25, .25, .80
MAX_SESIONES_SIN_CONTACTO, MAX_VIDA_DESDE_ORIGEN = 100, 220
PIVOTES_SEMILLA, MAX_SEMILLAS_POR_FECHA = 10, 2

def rma_pine(series, length):
    """
    Aproximación explícita a ta.rma() de Pine.

    Inicialización:
        SMA de los primeros 'length' valores válidos.

    Después:
        RMA_t =
            (RMA_(t-1) * (length - 1) + x_t)
            / length
    """

    s = pd.Series(series, copy=True).astype(float)

    values = s.to_numpy(dtype=float)

    out = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    valid_positions = np.flatnonzero(
        np.isfinite(values)
    )

    if len(valid_positions) < length:
        return pd.Series(
            out,
            index=s.index
        )

    seed_positions = valid_positions[:length]

    seed_pos = seed_positions[-1]

    seed = np.mean(
        values[seed_positions]
    )

    out[seed_pos] = seed

    prev = seed


    for i in range(
        seed_pos + 1,
        len(values)
    ):

        x = values[i]

        if not np.isfinite(x):

            out[i] = np.nan

            continue


        prev = (
            prev * (length - 1)
            +
            x
        ) / length

        out[i] = prev


    return pd.Series(
        out,
        index=s.index
    )

def calculate_rsi_pine(series, length):

    s = pd.Series(
        series,
        copy=True
    ).astype(float)


    delta = s.diff()


    gain = delta.clip(
        lower=0
    )


    loss = (
        -delta
    ).clip(
        lower=0
    )


    avg_gain = rma_pine(
        gain,
        length
    )


    avg_loss = rma_pine(
        loss,
        length
    )


    rsi = pd.Series(
        np.nan,
        index=s.index,
        dtype=float
    )


    normal = (
        (avg_loss > 0)
        &
        (avg_gain > 0)
    )


    rsi.loc[normal] = (
        100.0
        -
        100.0
        /
        (
            1.0
            +
            avg_gain.loc[normal]
            /
            avg_loss.loc[normal]
        )
    )


    solo_subidas = (
        (avg_loss == 0)
        &
        (avg_gain > 0)
    )

    rsi.loc[solo_subidas] = 100.0


    solo_bajadas = (
        (avg_gain == 0)
        &
        (avg_loss > 0)
    )

    rsi.loc[solo_bajadas] = 0.0


    plano = (
        (avg_gain == 0)
        &
        (avg_loss == 0)
    )

    rsi.loc[plano] = 50.0


    return rsi

def calculate_atr_pine(df, length=14):

    prev_close = (
        df["Close"]
        .shift(1)
    )


    tr = pd.concat(

        [

            df["High"]
            -
            df["Low"],


            (
                df["High"]
                -
                prev_close
            ).abs(),


            (
                df["Low"]
                -
                prev_close
            ).abs(),

        ],

        axis=1

    ).max(
        axis=1,
        skipna=True
    )


    return rma_pine(
        tr,
        length
    )

def pivot_low_pine_like(
    values,
    center,
    left,
    right
):

    if (
        center - left < 0
        or
        center + right >= len(values)
    ):

        return False


    c = values[center]


    if not np.isfinite(c):

        return False


    izquierda = values[
        center-left:center
    ]


    derecha = values[
        center+1:center+right+1
    ]


    if (
        not np.all(
            np.isfinite(izquierda)
        )
        or
        not np.all(
            np.isfinite(derecha)
        )
    ):

        return False


    return (
        np.all(
            c <= izquierda
        )
        and
        np.all(
            c < derecha
        )
    )

def buscar_referencia_alcista(
    pivots,
    curr_idx,
    curr_price,
    curr_rsi,
    curr_atr,
    params
):
    """
    Busca desde el pivote más reciente hacia atrás.

    Un mínimo antiguo sólo sigue siendo estructural si
    está por debajo de TODOS los mínimos intermedios.

    Se devuelve la PRIMERA divergencia válida.
    """

    if (
        not np.isfinite(curr_price)
        or
        not np.isfinite(curr_rsi)
        or
        not np.isfinite(curr_atr)
        or
        curr_atr <= 0
    ):

        return None


    min_bars = params["min_barras"]
    max_bars = params["max_barras"]

    min_rsi_diff = params["min_rsi_diff"]
    min_price_atr = params["min_price_atr"]

    nivel_os = params["nivel_sobreventa"]


    minimo_intermedio = None


    for ref in reversed(pivots):

        ref_idx = ref["idx"]

        ref_price = ref["price"]

        ref_rsi = ref["rsi"]


        barras_entre = (
            curr_idx
            -
            ref_idx
        )


        if barras_entre > max_bars:

            break


        referencia_estructural = (

            minimo_intermedio is None

            or

            ref_price
            <
            minimo_intermedio

        )


        if (
            referencia_estructural
            and
            barras_entre >= min_bars
            and
            np.isfinite(ref_rsi)
        ):

            diferencia_precio = (
                ref_price
                -
                curr_price
            )


            diferencia_rsi = (
                curr_rsi
                -
                ref_rsi
            )


            estructura_precio = (
                curr_price
                <
                ref_price
            )


            estructura_rsi = (
                diferencia_rsi
                >=
                min_rsi_diff
            )


            precio_ok = (
                diferencia_precio
                >=
                min_price_atr
                *
                curr_atr
            )


            sobreventa_ok = (
                ref_rsi <= nivel_os
                or
                curr_rsi <= nivel_os
            )


            divergencia_valida = (

                estructura_precio

                and

                estructura_rsi

                and

                precio_ok

                and

                sobreventa_ok

            )


            if divergencia_valida:

                return {

                    "ref_idx":
                        int(ref_idx),

                    "ref_price":
                        float(ref_price),

                    "ref_rsi":
                        float(ref_rsi),

                    "bars_between":
                        int(barras_entre),

                    "rsi_diff":
                        float(diferencia_rsi),

                    "price_atr":
                        float(
                            diferencia_precio
                            /
                            curr_atr
                        ),
                }


        if minimo_intermedio is None:

            minimo_intermedio = (
                ref_price
            )

        else:

            minimo_intermedio = min(
                minimo_intermedio,
                ref_price
            )


    return None

def crear_evento(
    direction,
    state,
    signal_idx,
    pivot_idx,
    curr_price,
    curr_rsi,
    referencia
):

    return {

        "direction":
            direction,

        "state":
            state,

        # ----------------------------------------------------
        # Día en que conocemos / vemos la señal
        # ----------------------------------------------------

        "signal_idx":
            int(signal_idx),

        # ----------------------------------------------------
        # Segundo extremo estructural
        # ----------------------------------------------------

        "pivot_idx":
            int(pivot_idx),

        # ----------------------------------------------------
        # Primer extremo estructural elegido por B estructural
        # ----------------------------------------------------

        "p1_idx":
            int(
                referencia["ref_idx"]
            ),

        "p1_price":
            float(
                referencia["ref_price"]
            ),

        "p2_price":
            float(
                curr_price
            ),

        "rsi1":
            float(
                referencia["ref_rsi"]
            ),

        "rsi2":
            float(
                curr_rsi
            ),

        "bars_between":
            int(
                referencia["bars_between"]
            ),

        "rsi_diff":
            float(
                referencia["rsi_diff"]
            ),

        "price_atr":
            float(
                referencia["price_atr"]
            ),
    }

def detectar(d, ticker):
    o,h,l,c,a = [d[k].to_numpy(dtype=float) for k in ('Open','High','Low','Close','ATR')]
    n=len(d)
    conf={}
    for i in range(L,n-R):
        wh=h[i-L:i+R+1]; wl=l[i-L:i+R+1]
        if h[i]==wh.max() and np.sum(wh==h[i])==1:
            conf.setdefault(i+R,[]).append({'tipo':'H','x':i,'y':float(h[i]),'confirmado':i+R})
        if l[i]==wl.min() and np.sum(wl==l[i])==1:
            conf.setdefault(i+R,[]).append({'tipo':'L','x':i,'y':float(l[i]),'confirmado':i+R})

    def geom(contactos,t):
        pts=sorted(contactos,key=lambda p:(p['x'],p['tipo']))
        altos=[p for p in pts if p['tipo']=='H']; bajos=[p for p in pts if p['tipo']=='L']
        if len(altos)<2 or len(bajos)<2: return None
        x0=min(p['x'] for p in pts); x1=max(p['x'] for p in pts); dur=x1-x0
        if not MIN_DURACION<=dur<=MAX_DURACION: return None
        for lado in (altos,bajos):
            if np.any(np.diff(sorted(p['x'] for p in lado))<MIN_SEP): return None
        if sum(p['tipo']!=q['tipo'] for p,q in zip(pts[:-1],pts[1:]))<3: return None
        mh,bh=np.polyfit([p['x'] for p in altos],[p['y'] for p in altos],1)
        ml,bl=np.polyfit([p['x'] for p in bajos],[p['y'] for p in bajos],1)
        if mh>=0 or ml>=0: return None
        ar=a[t]
        if not np.isfinite(ar) or ar<=0: return None
        if min(-mh*dur,-ml*dur)<DESCENSO_MIN_ATR*ar: return None
        if any(abs(p['y']-(mh*p['x']+bh))>TOL_PIVOTE_ATR*a[p['x']] for p in altos): return None
        if any(abs(p['y']-(ml*p['x']+bl))>TOL_PIVOTE_ATR*a[p['x']] for p in bajos): return None
        w0=(mh*x0+bh)-(ml*x0+bl); w1=(mh*x1+bh)-(ml*x1+bl)
        if min(w0,w1)<=ANCHURA_MIN_ATR*ar: return None
        contr=(w0-w1)/w0
        if CANAL_MIN_CONTRACCION<=contr<=CANAL_MAX_CONTRACCION: tipo='CANAL'
        elif CANAL_MAX_CONTRACCION<contr<CUNA_MAX_CONTRACCION and mh<ml: tipo='CUÑA'
        else: return None
        xx=np.arange(x0,t+1)
        band=TOL_CIERRE_ATR*a[xx]
        if not np.isfinite(band).all(): return None
        if np.any(c[xx]>(mh*xx+bh)+band): return None
        if np.any(c[xx]<(ml*xx+bl)-band): return None
        return dict(tipo=tipo,mh=float(mh),bh=float(bh),ml=float(ml),bl=float(bl),
                    x0=int(x0),x1=int(x1),duracion=int(dur),contraccion=float(contr),
                    nH=len(altos),nL=len(bajos),total=len(pts),equilibrio=min(len(altos),len(bajos)))

    def semillas(pivotes,t):
        rec=[p for p in pivotes if t-p['x']<=MAX_DURACION+R][-PIVOTES_SEMILLA:]
        if len(rec)<4: return []
        cand=[]
        for inds in combinations(range(len(rec)),4):
            pts=[rec[i] for i in inds]
            if not any(p['confirmado']==t for p in pts): continue
            g=geom(pts,t)
            if g is not None: cand.append((pts,g))
        if not cand: return []
        maxi=max(cand,key=lambda v:v[1]['duracion'])
        ans=[maxi]
        if MAX_SEMILLAS_POR_FECHA>1:
            mini=min(cand,key=lambda v:v[1]['duracion'])
            if mini!=maxi and mini[1]['duracion']<.75*maxi[1]['duracion']:
                ans.append(mini)
        return ans

    def dup(g,fs,t):
        for f in fs:
            if f['estado']!='ACTIVA': continue
            ratio=min(f['duracion'],g['duracion'])/max(f['duracion'],g['duracion'])
            if ratio<.80: continue
            if abs((f['mh']*t+f['bh'])-(g['mh']*t+g['bh']))<.5*a[t] and \
               abs((f['ml']*t+f['bl'])-(g['ml']*t+g['bl']))<.5*a[t]:
                return True
        return False

    pivotes=[]; figuras=[]; rupturas=[]; nextid=1
    for t in range(max(ATR_N+L+R+5,30),n):
        nuevos=conf.get(t,[])
        # Evaluar señales ANTES de actualizar geometria con pivotes confirmados hoy.
        for f in figuras:
            if f['estado']!='ACTIVA' or t<=f['creada']: continue
            th=f['mh']*t+f['bh']; ty=f['mh']*(t-1)+f['bh']
            arriba=(c[t]>th+TOL_CIERRE_ATR*a[t-1] and c[t-1]<=ty+TOL_CIERRE_ATR*a[t-2])
            if arriba:
                f['estado']='ROTA_AL_ALZA'; f['fin']=t
                rupturas.append(dict(Ticker=ticker, ID=f['id'], Tipo=f['tipo'],t=t,
                     Fecha=d.index[t], Entrada_fecha=d.index[t+1] if t+1<n else pd.NaT,
                     nH=f['nH'], nL=f['nL'],Contactos=f['total'],Equilibrio=f['equilibrio'],
                     Duracion=f['duracion'],Actualizaciones=f['actualizaciones'],
                     Contraccion=f['contraccion'],Ancla_inicio=d.index[f['x0']],
                     Techo_ruptura=th))
                continue
            if c[t]<f['ml']*t+f['bl']-TOL_CIERRE_ATR*a[t-1]:
                f['estado']='ROTA_ABAJO'; f['fin']=t
            elif t-f['ultimo_contacto']>MAX_SESIONES_SIN_CONTACTO or t-f['x0']>MAX_VIDA_DESDE_ORIGEN:
                f['estado']='CADUCADA'; f['fin']=t
        if nuevos:
            pivotes.extend(nuevos)
            for f in figuras:
                if f['estado']!='ACTIVA': continue
                for p in nuevos:
                    if p['x']<=f['x1'] or any(q['tipo']==p['tipo'] and q['x']==p['x'] for q in f['contactos']): continue
                    nivel=(f['mh']*p['x']+f['bh']) if p['tipo']=='H' else (f['ml']*p['x']+f['bl'])
                    if abs(p['y']-nivel)>TOL_PIVOTE_ATR*a[p['x']]: continue
                    propuesta=f['contactos']+[p]; g=geom(propuesta,t)
                    if g is None: continue
                    f['contactos']=propuesta; f.update(g)
                    f['ultimo_contacto']=t; f['actualizaciones']+=1
            for pts,g in semillas(pivotes,t):
                if dup(g,figuras,t): continue
                f=dict(id=nextid,estado='ACTIVA',creada=t,fin=None,ultimo_contacto=t,
                       actualizaciones=0,contactos=list(pts))
                f.update(g); figuras.append(f); nextid+=1
    # Una señal principal por activo y dia. La geometria se conserva en las columnas.
    principales=[]
    for t in sorted(set(s['t'] for s in rupturas)):
        grupo=[s for s in rupturas if s['t']==t]
        principales.append(max(grupo,key=lambda s:(s['Contactos'],s['Equilibrio'],s['Duracion'])))
    return principales,figuras,rupturas

def detectar_divergencias_alcistas(df, rsi14, atr14, params):
    precio = (df['Open'].to_numpy(dtype=float) + df['High'].to_numpy(dtype=float) +
              df['Low'].to_numpy(dtype=float) + df['Close'].to_numpy(dtype=float)) / 4.0
    rsi = rsi14.to_numpy(dtype=float)
    atr = atr14.to_numpy(dtype=float)
    n = len(df)
    left, right = params['pivot_left'], params['pivot_right']
    max_bars = params['max_barras']
    low_pivots, confirmed = [], []
    bull_ghost = None

    for i in range(n):
        center = i-right
        low_pivot_confirmed = (
            center >= left and
            pivot_low_pine_like(precio, center, left, right)
        )
        if low_pivot_confirmed:
            while low_pivots and center-low_pivots[0]['idx'] > max_bars:
                low_pivots.pop(0)
            ref = buscar_referencia_alcista(
                low_pivots, center, precio[center], rsi[center], atr[center], params
            )
            if ref is not None:
                confirmed.append(crear_evento(
                    direction='ALCISTA', state='CONFIRMADA', signal_idx=i,
                    pivot_idx=center, curr_price=precio[center],
                    curr_rsi=rsi[center], referencia=ref
                ))
            low_pivots.append(dict(
                idx=int(center), price=float(precio[center]),
                rsi=float(rsi[center]) if np.isfinite(rsi[center]) else np.nan
            ))

        if (params['mostrar_fantasmas'] and np.isfinite(precio[i]) and
            np.isfinite(rsi[i]) and np.isfinite(atr[i])):
            ref = buscar_referencia_alcista(
                low_pivots, i, precio[i], rsi[i], atr[i], params
            )
            potencial = ref is not None
            if bull_ghost is not None and i-bull_ghost['p1_idx'] > max_bars:
                bull_ghost = None
            if potencial and bull_ghost is None:
                bull_ghost = crear_evento(
                    direction='ALCISTA', state='FANTASMA', signal_idx=i,
                    pivot_idx=i, curr_price=precio[i], curr_rsi=rsi[i], referencia=ref
                )
            elif bull_ghost is not None and precio[i] < bull_ghost['p2_price']:
                bull_ghost = (
                    crear_evento(direction='ALCISTA', state='FANTASMA',
                                 signal_idx=i, pivot_idx=i,
                                 curr_price=precio[i], curr_rsi=rsi[i], referencia=ref)
                    if potencial else None
                )
        if low_pivot_confirmed:
            bull_ghost = None

    ghosts = [bull_ghost.copy()] if bull_ghost is not None and params['mostrar_fantasmas'] else []
    return confirmed, ghosts

EVENT_COLUMNS = ['Senal','Estado','Fecha','Velas_hace','signal_idx','pivot_idx','Precio_cierre']
LABELS = {'DIV_ALCISTA':'🟠 Divergencia', 'CUNA_ROTA':'🟢 Cuña rota',
          'RSI_AZUL':'🔵 RSI65/8', 'DIV_FANTASMA':'🟠👻 Fantasma'}

def divergence_frame(df):
    rsi = calculate_rsi_pine(df.Close, 14)
    confirmed, ghosts = detectar_divergencias_alcistas(df,rsi,calculate_atr_pine(df),PARAMS)
    result = pd.DataFrame({'divergencia_tipo':'ninguna'}, index=df.index)
    for ev in confirmed:
        result.iloc[ev['signal_idx'],0]='alcista'
    return result

def structural_data(df, ticker=''):
    d=df.copy()
    rsi14=calculate_rsi_pine(d.Close,14)
    confirmed, ghosts=detectar_divergencias_alcistas(d,rsi14,calculate_atr_pine(d),PARAMS)
    rsi65=calculate_rsi_pine((d.Open+d.High+d.Low+d.Close)/4,65)
    media65=rsi65.rolling(8).mean()
    slope=media65-media65.shift(2)
    blue=((rsi65>45)&(rsi65.shift(1)<=45)&(slope>=0.35)).fillna(False)
    pc=d.Close.shift(1)
    tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean()
    main, figures, _=detectar(d,ticker)
    wedges=[s for s in main if s['Tipo']=='CUÑA']
    events=[]
    def add(kind,t,state='CONFIRMADA',pivot=None):
        events.append(dict(Senal=kind,Estado=state,Fecha=d.index[t],Velas_hace=len(d)-1-t,
                           signal_idx=t,pivot_idx=t if pivot is None else pivot,Precio_cierre=float(d.Close.iloc[t])))
    for ev in confirmed: add('DIV_ALCISTA',ev['signal_idx'],pivot=ev['pivot_idx'])
    for ev in ghosts: add('DIV_FANTASMA',ev['signal_idx'],'FANTASMA',ev['pivot_idx'])
    for t in np.flatnonzero(blue.to_numpy()): add('RSI_AZUL',int(t))
    for ev in wedges: add('CUNA_ROTA',ev['t'])
    frame=pd.DataFrame(events,columns=EVENT_COLUMNS)
    frame['Cambio_desde_senal_pct']=(float(d.Close.iloc[-1])/frame.Precio_cierre-1)*100
    return dict(events=frame,rsi14=rsi14,rsi65=rsi65,
                media65=media65,slope=slope,figures=figures,wedges=wedges,atr=d.ATR)

def scanner_table(events, names, window=25):
    rows=[]
    if events.empty: return pd.DataFrame()
    recent=events[events.Velas_hace<=window]
    for ticker,g in recent.groupby('Ticker'):
        row={'Ticker':ticker,'Nombre':names.get(ticker,ticker)}
        ages=[]; ghost_age=None
        for key,label in LABELS.items():
            subset=g[g.Senal==key].sort_values('Velas_hace')
            row[label]='—'
            row['_'+key]=not subset.empty
            if not subset.empty:
                e=subset.iloc[0]; age=int(e.Velas_hace)
                row[label]=f"{e.Fecha:%Y-%m-%d} · {age}v"
                if key!='DIV_FANTASMA': ages.append(age)
                else: ghost_age=age
        confirmed=g[g.Senal!='DIV_FANTASMA'].sort_values('Velas_hace')
        row['Desde última señal (%)']=round(float(confirmed.iloc[0]['Cambio_desde_senal_pct']),2) if not confirmed.empty and 'Cambio_desde_senal_pct' in confirmed else None
        row['Coincidencias']=len(ages)
        row['Reciente (velas)']=min(ages) if ages else None
        row['_ghost_age']=ghost_age
        rows.append(row)
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['Reciente (velas)','Coincidencias','_ghost_age','Ticker'],
                                          ascending=[True,False,True,True],na_position='last')
