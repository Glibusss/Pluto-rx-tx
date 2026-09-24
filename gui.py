from __future__ import annotations
import codecs
import json
import math
import queue
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from modem import Config, MODS, frame_len, PREAMBLE
from session import Source
from radio import (RX_QUEUE_DEFAULT, calibrate_noise, discover_pluto_usb, receive,
                   rx_queue_capacity, transmit)


APP_ICON_SIZE=32
APP_ICON_BACKGROUND='#202124'
APP_ICON_SIGNAL='#66d17a'


def app_icon_pixels(size=APP_ICON_SIZE):
    """Return a deterministic radio-signal icon; no external icon file needed."""
    if size<16:
        raise ValueError('Размер иконки должен быть не меньше 16 пикселей')
    cx=(size-1)/2
    cy=size*.72
    scale=size/32
    pixels=[]
    for y in range(size):
        row=[]
        for x in range(size):
            distance=math.hypot(x-cx,y-cy)
            arcs=(y<cy-2*scale and
                  (abs(distance-7*scale)<=1.05*scale or
                   abs(distance-12*scale)<=1.05*scale))
            mast=abs(x-cx)<=1.05*scale and cy-6*scale<=y<=cy
            source=distance<=2.35*scale
            row.append(APP_ICON_SIGNAL if arcs or mast or source else APP_ICON_BACKGROUND)
        pixels.append(tuple(row))
    return tuple(pixels)


def make_app_icon(master):
    image=tk.PhotoImage(master=master,width=APP_ICON_SIZE,height=APP_ICON_SIZE)
    for y,row in enumerate(app_icon_pixels()):
        image.put('{'+ ' '.join(row) +'}',to=(0,y))
    return image


def gain_power_text(value):
    """Describe an AD936x gain value as an ideal linear power ratio."""
    try:
        db=float(str(value).strip().replace(',','.'))
        if not math.isfinite(db):
            raise ValueError
        ratio=10**(db/10)
    except (TypeError,ValueError,OverflowError):
        return 'линейно по мощности: —'
    number=f'{ratio:.4g}'.replace('.',',')
    return f'линейно по мощности: ×{number}'


class App(tk.Tk):
    def __init__(self, tx):
        super().__init__()
        self.app_icon=make_app_icon(self)
        self.iconphoto(True,self.app_icon)
        self.tx=tx
        self.title('PlutoSDR — '+('Передатчик' if tx else 'Приёмник / BER / PER'))
        self.geometry('1200x850')
        self.minsize(900,700)
        self.events=queue.SimpleQueue()
        self.latest={}
        self.latest_lock=threading.Lock()
        self.stop_event=threading.Event()
        self.worker=None
        self.discovery_worker=None
        self.closing=False
        self.source=None
        self.photo_refs={}
        self.controls=[]
        self.output=None
        self.received_transfer=None
        self.received_bytes=0
        self.received_decoder=None
        self.received_final=False
        self.received_limited=False
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.build()
        self.after(100,self.poll)
        self.after(250,lambda:self.find_pluto(automatic=True))

    def build(self):
        main=ttk.Frame(self,padding=10)
        main.pack(fill='both',expand=True)
        rf=ttk.LabelFrame(main,text='Параметры радио — должны совпадать на TX и RX',padding=8)
        rf.pack(fill='x')
        self.uri=tk.StringVar(value='ip:192.168.2.1')
        self.freq=tk.StringVar(value='2400')
        self.mod=tk.StringVar(value='BPSK')
        self.rate=tk.StringVar(value='1')
        self.gain=tk.StringVar(value='-30' if self.tx else '20')
        self.gain_info=tk.StringVar(value=gain_power_text(self.gain.get()))
        self.cfo=tk.StringVar(value='40000')
        self.gap=tk.StringVar(value='20')
        self.queue_buffers=tk.StringVar(value=str(RX_QUEUE_DEFAULT))
        self.squelch=tk.BooleanVar(value=False)
        self.squelch_threshold=tk.StringVar(value='')
        self.pre=tk.BooleanVar(value=True)
        self.cp=tk.BooleanVar(value=False)
        entries=[('URI Pluto',self.uri,24),('Частота центра, МГц',self.freq,13),
                 ('TX gain Pluto, dB (≤0)' if self.tx else 'RX gain Pluto, dB',self.gain,12)]
        for col,(label,var,width) in enumerate(entries):
            ttk.Label(rf,text=label).grid(row=0,column=col*2,sticky='w',padx=4)
            parent=rf
            if var is self.uri or var is self.gain:
                parent=ttk.Frame(rf)
                parent.grid(row=0,column=col*2+1,padx=4,sticky='ew')
            if var is self.uri:
                w=ttk.Combobox(parent,textvariable=var,width=width,state='normal')
                w.pack(side='left',fill='x',expand=True)
                self.uri_widget=w
                self.discovery_button=ttk.Button(parent,text='Найти USB',command=self.find_pluto)
                self.discovery_button.pack(side='left',padx=(7,0))
                self.controls.append((self.discovery_button,'normal'))
            else:
                w=ttk.Entry(parent,textvariable=var,width=width)
            if var is self.gain:
                w.pack(side='left')
                ttk.Label(parent,textvariable=self.gain_info).pack(side='left',padx=(7,0))
            elif var is not self.uri:
                w.grid(row=0,column=col*2+1,padx=4,sticky='ew')
            self.controls.append((w,'normal'))
        self.gain.trace_add('write',self.update_gain_info)
        ttk.Label(rf,text='Модуляция').grid(row=1,column=0,sticky='w',padx=4,pady=8)
        w=ttk.Combobox(rf,textvariable=self.mod,values=MODS,state='readonly',width=20)
        w.grid(row=1,column=1,sticky='ew',padx=4)
        self.controls.append((w,'readonly'))
        ttk.Label(rf,text='Sample rate, MS/s').grid(row=1,column=2,sticky='w',padx=4)
        w=ttk.Combobox(rf,textvariable=self.rate,values=('1','2'),state='readonly',width=10)
        w.grid(row=1,column=3,sticky='ew',padx=4)
        self.controls.append((w,'readonly'))
        label='Пауза после буфера, мс' if self.tx else 'Поиск CFO ± Гц'
        ttk.Label(rf,text=label).grid(row=1,column=4,sticky='w',padx=4)
        w=ttk.Entry(rf,textvariable=self.gap if self.tx else self.cfo,width=12)
        w.grid(row=1,column=5,sticky='ew',padx=4)
        self.controls.append((w,'normal'))
        options=ttk.Frame(rf)
        options.grid(row=2,column=0,columnspan=6,sticky='w')
        for label,var in [('Преамбула',self.pre),('Циклический префикс: 16 символов',self.cp)]:
            w=ttk.Checkbutton(options,text=label,variable=var)
            w.pack(side='left',padx=5)
            self.controls.append((w,'normal'))
        if not self.tx:
            ttk.Label(options,text='IQ очередь, буферов:').pack(side='left',padx=(12,4))
            w=ttk.Entry(options,textvariable=self.queue_buffers,width=7)
            w.pack(side='left')
            self.controls.append((w,'normal'))
        ttk.Label(options,text='Заголовок BPSK; RGB-блок 16×16; пилоты в каждом блоке').pack(side='left',padx=12)
        if not self.tx:
            squelch=ttk.Frame(rf)
            squelch.grid(row=3,column=0,columnspan=6,sticky='w',pady=(7,0))
            w=ttk.Checkbutton(squelch,text='Отсекать шум',variable=self.squelch)
            w.pack(side='left',padx=5)
            self.controls.append((w,'normal'))
            ttk.Label(squelch,text='Порог, dBFS:').pack(side='left',padx=(8,4))
            w=ttk.Entry(squelch,textvariable=self.squelch_threshold,width=9)
            w.pack(side='left')
            self.controls.append((w,'normal'))
            w=ttk.Button(squelch,text='Калибровка шума',command=self.calibrate)
            w.pack(side='left',padx=8)
            self.controls.append((w,'normal'))
            ttk.Label(squelch,text='Перед калибровкой выключите TX').pack(side='left',padx=4)
        src=ttk.LabelFrame(main,text='Данные для передачи' if self.tx else 'Эталон — тот же файл или точно тот же текст, что на TX',padding=8)
        src.pack(fill='both',expand=True,pady=8)
        self.kind=tk.StringVar(value='image')
        top=ttk.Frame(src)
        top.pack(fill='x')
        for value,label in [('image','PNG / JPG → RGB'),('text','Текст UTF-8')]:
            w=ttk.Radiobutton(top,text=label,value=value,variable=self.kind)
            w.pack(side='left',padx=4)
            self.controls.append((w,'normal'))
        w=ttk.Button(top,text='Выбрать изображение',command=self.load_image)
        w.pack(side='left',padx=8)
        self.controls.append((w,'normal'))
        w=ttk.Button(top,text='Загрузить TXT (необязательно)',command=self.load_text)
        w.pack(side='left',padx=4)
        self.controls.append((w,'normal'))
        self.source_label=ttk.Label(top,text='Файл не выбран')
        self.source_label.pack(side='left',padx=8)
        ttk.Label(src,text='Или введите текст прямо в поле:').pack(anchor='w',pady=(8,0))
        text_area=ttk.Frame(src)
        text_area.pack(fill='both',expand=True,pady=(3,0))
        text_area.columnconfigure(0,weight=1)
        text_area.rowconfigure(0,weight=1)
        self.text=tk.Text(text_area,height=8,wrap='word',font=('Consolas',10),undo=True)
        text_scroll=ttk.Scrollbar(text_area,orient='vertical',command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.grid(row=0,column=0,sticky='nsew')
        text_scroll.grid(row=0,column=1,sticky='ns')
        self.text.bind('<FocusIn>',self.select_text_input)
        self.controls.append((self.text,'normal'))
        run=ttk.Frame(main)
        run.pack(fill='x')
        self.start_btn=ttk.Button(run,text='Start TX' if self.tx else 'Start RX',command=self.start)
        self.start_btn.pack(side='left')
        self.stop_btn=ttk.Button(run,text='Stop',command=self.stop,state='disabled')
        self.stop_btn.pack(side='left',padx=8)
        self.progress=ttk.Progressbar(run,mode='determinate')
        self.progress.pack(side='left',fill='x',expand=True,padx=8)
        self.status=ttk.Label(run,text='Остановлено')
        self.status.pack(side='left',padx=8)
        self.folder=tk.StringVar(value=str(Path.cwd()/'received'))
        if not self.tx:
            dst=ttk.Frame(main)
            dst.pack(fill='x',pady=(8,0))
            ttk.Label(dst,text='Сохранение результатов:').pack(side='left')
            w=ttk.Entry(dst,textvariable=self.folder)
            w.pack(side='left',fill='x',expand=True,padx=8)
            self.controls.append((w,'normal'))
            w=ttk.Button(dst,text='Папка…',command=self.pick_folder)
            w.pack(side='left')
            self.controls.append((w,'normal'))
        self.metrics=ttk.Label(main,text='BER: —    PER: —    SNR: —',font=('Segoe UI',11),wraplength=1140)
        self.metrics.pack(fill='x',pady=8)
        self.tabs=ttk.Notebook(main)
        self.tabs.pack(fill='both',expand=True)
        image_tab=ttk.Frame(self.tabs)
        self.tabs.add(image_tab,text='Изображение / текст')
        self.left=ttk.Label(image_tab,text='Исходное изображение / эталон',anchor='center')
        self.left.pack(side='left',fill='both',expand=True)
        self.right=ttk.Label(image_tab,text='Принятое изображение',anchor='center')
        if not self.tx:
            self.right.pack(side='left',fill='both',expand=True)
        received_tab=ttk.Frame(self.tabs)
        received_tab.columnconfigure(0,weight=1)
        received_tab.rowconfigure(0,weight=1)
        self.received_text=tk.Text(received_tab,wrap='word',state='disabled',font=('Consolas',11))
        received_scroll=ttk.Scrollbar(received_tab,orient='vertical',command=self.received_text.yview)
        self.received_text.configure(yscrollcommand=received_scroll.set)
        self.received_text.grid(row=0,column=0,sticky='nsew')
        received_scroll.grid(row=0,column=1,sticky='ns')
        self.tabs.add(received_tab,text='Принятый текст')
        self.plot=tk.Canvas(self.tabs,bg='white',highlightthickness=0)
        self.tabs.add(self.plot,text='Решения демодулятора')
        self.packet_canvas=tk.Canvas(self.tabs,bg='white',highlightthickness=0)
        self.tabs.add(self.packet_canvas,text='Пакеты')
        self.log=tk.Text(self.tabs,height=8,wrap='word',state='disabled',font=('Consolas',10))
        self.tabs.add(self.log,text='Лог')
        self.plot.bind('<Configure>',lambda e:self.draw_plots())
        self.packet_canvas.bind('<Configure>',lambda e:self.draw_plots())
        self.write_log('Первым запустите RX. Затем отправьте данные с TX. Настройки фиксируются до Stop.')

    def load_image(self):
        path=filedialog.askopenfilename(filetypes=[('Изображения','*.png *.jpg *.jpeg'),('Все файлы','*.*')])
        if not path:return
        try:
            self.source=Source.image(path)
            self.kind.set('image')
            self.source_label.configure(text=(f'{self.source.label}: {self.source.width}×{self.source.height}, '
                f'{self.source.total} пакетов, SHA {self.source.digest.hex()[:8]}'))
            self.show_image(self.left,Image.fromarray(self.source.rgb),'left')
        except Exception as e:messagebox.showerror('Изображение',str(e))

    def update_gain_info(self,*_):
        self.gain_info.set(gain_power_text(self.gain.get()))

    def load_text(self):
        path=filedialog.askopenfilename(filetypes=[('Текст','*.txt'),('Все файлы','*.*')])
        if not path:return
        try:
            value=Path(path).read_text(encoding='utf-8')
            self.text.delete('1.0','end')
            self.text.insert('1.0',value)
            self.kind.set('text')
            self.source_label.configure(text=f'{Path(path).name}: текст загружен')
        except Exception as e:messagebox.showerror('Текст',str(e))

    def select_text_input(self,event=None):
        self.kind.set('text')
        self.source_label.configure(text='Текст из поля — файл не требуется')

    def pick_folder(self):
        path=filedialog.askdirectory()
        if path:self.folder.set(path)

    def find_pluto(self,automatic=False):
        if self.discovery_worker and self.discovery_worker.is_alive():
            return
        initial_uri=self.uri.get()
        self.discovery_button.configure(state='disabled')
        if not automatic:
            self.write_log('Поиск Pluto среди USB-контекстов libiio…')
        def work():
            try:
                found=discover_pluto_usb()
                self.events.put(('discovery',(found,initial_uri,automatic)))
            except Exception as error:
                self.events.put(('discovery_error',(str(error),automatic)))
        self.discovery_worker=threading.Thread(target=work,name='Pluto USB discovery',daemon=True)
        self.discovery_worker.start()

    def show_image(self,label,im,key):
        im=im.copy()
        im.thumbnail((530,410))
        self.photo_refs[key]=ImageTk.PhotoImage(im)
        label.configure(image=self.photo_refs[key],text='')

    def get_source(self):
        if self.kind.get()=='image':return self.source
        value=self.text.get('1.0','end-1c')
        return Source.text(value) if value else None

    def reset_received_text(self,transfer=None):
        self.received_transfer=transfer
        self.received_bytes=0
        self.received_decoder=(codecs.getincrementaldecoder('utf-8')(errors='replace')
                               if transfer is not None else None)
        self.received_final=False
        self.received_limited=False
        self.received_text.configure(state='normal')
        self.received_text.delete('1.0','end')
        self.received_text.configure(state='disabled')

    def settings(self,include_squelch=True):
        cfg=Config(self.mod.get(),self.pre.get(),self.cp.get(),int(float(self.rate.get())*1e6),float(self.cfo.get()))
        cfg.validate()
        gain=float(self.gain.get())
        if not (-89.75<=gain<=0 if self.tx else -3<=gain<=71):
            raise ValueError('TX gain: -89.75…0 dB; RX gain: -3…71 dB')
        freq=int(float(self.freq.get().replace(',','.'))*1e6)
        if not 326e6<=freq<=3800e6:raise ValueError('Частота: 326…3800 МГц')
        gap=float(self.gap.get())
        if not 0<=gap<=2000:raise ValueError('Пауза: 0…2000 мс')
        queue_buffers=rx_queue_capacity({'queue_buffers':self.queue_buffers.get()})
        squelch_dbfs=None
        if not self.tx and include_squelch and self.squelch.get():
            if not self.squelch_threshold.get().strip():
                raise ValueError('Сначала выполните калибровку шума или введите порог dBFS')
            squelch_dbfs=float(self.squelch_threshold.get().replace(',','.'))
            if not -150<=squelch_dbfs<=10:
                raise ValueError('Шумовой порог: -150…10 dBFS')
        if not self.uri.get().strip():raise ValueError('Введите URI Pluto')
        return dict(cfg=cfg,frequency=freq,gain=gain,gap_ms=gap,uri=self.uri.get().strip(),
                    queue_buffers=queue_buffers,squelch_dbfs=squelch_dbfs)

    def calibrate(self):
        if self.tx or self.worker and self.worker.is_alive():return
        try:
            settings=self.settings(include_squelch=False)
        except Exception as e:
            messagebox.showerror('Параметры',str(e));return
        self.stop_event.clear()
        for widget,state in self.controls:widget.configure(state='disabled')
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.status.configure(text='Калибровка шума…')
        self.write_log('Калибровка шума: TX должен быть выключен. Измерение около 2 секунд.')
        def work():
            try:
                result=calibrate_noise(settings,self.stop_event)
                if result is not None:self.emit('calibration',result)
            except Exception:
                self.emit('error',traceback.format_exc())
            finally:self.emit('done',None)
        self.worker=threading.Thread(target=work,name='Pluto noise calibration',daemon=True)
        self.worker.start()

    def start(self):
        if self.worker and self.worker.is_alive():return
        try:
            settings=self.settings()
            source=self.get_source()
            if self.tx and source is None:raise ValueError('Выберите изображение или введите текст')
            folder=self.folder.get()
            if not self.tx:
                Path(folder).mkdir(parents=True,exist_ok=True)
        except Exception as e:
            messagebox.showerror('Параметры',str(e));return
        self.stop_event.clear()
        self.output=None
        if not self.tx:self.reset_received_text()
        for widget,state in self.controls:widget.configure(state='disabled')
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.status.configure(text='Подключение…')
        self.progress.configure(value=0,maximum=source.total if source else 1)
        cfg=settings['cfg']
        rs=cfg.sample_rate/8
        self.write_log(f'{cfg.mod}; Rs={rs:g} симв/с; preamble={cfg.preamble}; CP={cfg.cp}.')
        if not self.tx and settings['squelch_dbfs'] is not None:
            self.write_log(f'Отсечение шума включено; порог {settings["squelch_dbfs"]:.1f} dBFS.')
        if self.tx:
            duration=(frame_len(cfg)+(len(PREAMBLE) if cfg.preamble else 0))/cfg.sample_rate
            self.write_log(f'Радиопакет ≈{duration*1000:.1f} мс. Одинаковая Rs и средняя энергия символа; не одинаковая Eb/N0.')
        def work():
            try:
                if self.tx:transmit(settings,source,self.stop_event,self.emit)
                else:receive(settings,source,folder,self.stop_event,self.emit)
            except Exception:
                self.emit('error',traceback.format_exc())
            finally:self.emit('done',None)
        self.worker=threading.Thread(target=work,name='Pluto task',daemon=True)
        self.worker.start()

    def emit(self,kind,value):
        if kind in ('snapshot','health','progress'):
            with self.latest_lock:self.latest[kind]=value
        else:self.events.put((kind,value))

    def stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state='disabled')
        self.status.configure(text='Остановка…')

    def close(self):
        self.closing=True
        if self.worker and self.worker.is_alive():self.stop()
        else:self.destroy()

    def write_log(self,value):
        self.log.configure(state='normal')
        self.log.insert('end',time.strftime('%H:%M:%S')+' '+value+'\n')
        if int(self.log.index('end-1c').split('.')[0])>2000:self.log.delete('1.0','501.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        with self.latest_lock:
            latest=self.latest
            self.latest={}
        for kind,value in latest.items():
            if kind=='snapshot':self.update_snapshot(value)
            elif kind=='progress':
                self.progress.configure(value=value[0],maximum=value[1])
                cycle=value[2] if len(value)>2 else 1
                self.status.configure(text=f'TX · цикл {cycle} · {value[0]}/{value[1]}')
            elif kind=='health' and not self.stop_event.is_set():
                capacity=value.get('queue_capacity','?')
                status=f'RX · IQ очередь {value["queue"]}/{capacity} · потери {value["drops"]}'
                if value.get('squelch_dbfs') is not None:
                    level=value.get('input_dbfs')
                    status+=f' · вход {"—" if level is None else f"{level:.1f}"} dBFS · отсечено {value["squelched"]}'
                self.status.configure(text=status)
        for _ in range(100):
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind in ('log','error'):
                self.write_log(value)
                if kind=='error':self.tabs.select(self.log)
            elif kind=='calibration':
                # CFAR margins can be fractions of a decibel; keep enough precision
                # when transferring the calibrated threshold into the editable field.
                self.squelch_threshold.set(f'{value["threshold_dbfs"]:.2f}')
                self.squelch.set(True)
                self.write_log(f'Калибровка завершена: шум {value["noise_dbfs"]:.1f} dBFS, '+
                    f'σ мощности {value["spread_db"]:.2f} dB, порог {value["threshold_dbfs"]:.1f} dBFS '+
                    f'(P_FA≤{value["false_alarm_probability"]:.0e}), '
                    f'буферов {value["used_buffers"]}/{value["buffers"]}, '
                    f'выбросов {value["outliers"]}.')
            elif kind=='discovery':
                found,initial_uri,automatic=value
                self.discovery_worker=None
                uris=[item['uri'] for item in found]
                self.uri_widget.configure(values=uris)
                running=self.worker and self.worker.is_alive()
                if not running:
                    self.discovery_button.configure(state='normal')
                if found:
                    confirmed=next((item for item in found if item['identified']),None)
                    # Never overwrite text typed while a scan was running.
                    if confirmed and not running and self.uri.get()==initial_uri:
                        self.uri.set(confirmed['uri'])
                    selected=confirmed or found[0]
                    description=f' — {selected["description"]}' if selected['description'] else ''
                    suffix=(f'; всего USB-контекстов: {len(found)}. Выберите URI из списка.'
                            if len(found)>1 else '.')
                    label=('Pluto USB' if confirmed else
                           'USB IIO-контекст (описание не подтверждает Pluto)')
                    self.write_log(f'Найден {label}: {selected["uri"]}{description}{suffix}')
                else:
                    self.write_log('Pluto через USB backend не найден; ручной URI оставлен без изменений.')
            elif kind=='discovery_error':
                error,automatic=value
                self.discovery_worker=None
                if not (self.worker and self.worker.is_alive()):
                    self.discovery_button.configure(state='normal')
                self.write_log(('Автопоиск' if automatic else 'Поиск USB')+': '+error+
                               '. Ручной URI оставлен без изменений.')
            elif kind=='done':
                for widget,state in self.controls:widget.configure(state=state)
                if self.discovery_worker and self.discovery_worker.is_alive():
                    self.discovery_button.configure(state='disabled')
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                self.status.configure(text='Остановлено')
                if self.closing:self.destroy();return
        self.after(100,self.poll)

    def update_snapshot(self,snapshot):
        self.output=snapshot
        s=snapshot['stats']
        number=lambda x:'—' if x is None else f'{x:.5g}'
        self.metrics.configure(text=(f'BER: {number(s["ber"])}  (проверено {s["ber_coverage"]:.1%} бит)     '
            f'PER: {s["per"]:.4f}'+(' [итог]' if s['per_final'] else ' [предварительно]')+
            f'     SNR-оценка: {number(s["snr_estimate_db"])} dB\n'
            f'GOOD: {s["good_packets"]}   CRC: {s["crc_packets"]}   '
            f'{"LOST" if s["per_final"] else "Ожидаются / потеряны"}: {s["missing_packets"]}   '
            f'Всего: {s["expected_packets"]}   CFO: {s["last_cfo_hz"]:.0f} Гц'+
            ('' if s['reference_match'] else f'\nBER недоступен: {s["reference_status"]}')))
        self.progress.configure(maximum=s['expected_packets'],value=s['received_packets'])
        if s['kind']=='RGB':
            self.show_image(self.right,Image.frombytes('RGB',(s['width'],s['height']),snapshot['data']),'right')
        else:
            # Append newly received bytes immediately. The incremental decoder keeps
            # UTF-8 characters intact when a packet boundary falls inside a character.
            transfer=s['transfer']
            preview_size=min(snapshot['preview_size'],200000)
            if (transfer!=self.received_transfer or preview_size<self.received_bytes or
                    (self.received_final and preview_size>self.received_bytes)):
                self.reset_received_text(transfer)
            chunk=snapshot['data'][self.received_bytes:preview_size]
            final=s['per_final'] and not self.received_final
            text=self.received_decoder.decode(chunk,final=final).replace('\0','□')
            self.received_text.configure(state='normal')
            if text:self.received_text.insert('end',text)
            if snapshot['preview_size']>200000 and not self.received_limited:
                self.received_text.insert('end','\n[Предпросмотр ограничен 200000 байтами]')
                self.received_limited=True
            self.received_text.see('end')
            self.received_text.configure(state='disabled')
            self.received_bytes=preview_size
            self.received_final=self.received_final or final
        self.draw_plots()

    def draw_plots(self):
        if not self.output:return
        c=self.plot;c.delete('all')
        w,h=max(c.winfo_width(),200),max(c.winfo_height(),150)
        sample=np.asarray(self.output['samples'])
        if sample.size:
            fsk=self.output['stats']['modulation']=='2-FSK'
            span=max(1.6,float(np.quantile(abs(sample),.98))*1.2)
            xmin,xmax=(-.2,span) if fsk else (-span,span)
            ymin,ymax=(-.2,span) if fsk else (-span,span)
            sx=lambda v:35+(v-xmin)/(xmax-xmin)*(w-70)
            sy=lambda v:h-30-(v-ymin)/(ymax-ymin)*(h-60)
            c.create_line(sx(0),20,sx(0),h-20,fill='#aaa')
            c.create_line(20,sy(0),w-20,sy(0),fill='#aaa')
            for x,y in sample:
                px,py=sx(x),sy(y)
                c.create_oval(px-2,py-2,px+2,py+2,fill='#1765a5',outline='')
            c.create_text(15,15,anchor='nw',text='|коррелятор f0| / |коррелятор f1|' if fsk else 'I / Q после коррекции по пилотам')
        c=self.packet_canvas;c.delete('all')
        w=max(c.winfo_width(),200)
        total=self.output['stats']['expected_packets']
        statuses=self.output['statuses']
        c.create_text(15,15,anchor='nw',text='Зелёный: GOOD · оранжевый: CRC · серый: не принят. Одна ячейка может объединять пакеты.')
        cols=max(1,(w-30)//12)
        count=min(total,cols*20)
        for i in range(count):
            a,b=i*total//count,(i+1)*total//count
            group=[statuses.get(n,'LOST') for n in range(a,b)]
            color='#b0b0b0' if 'LOST' in group else '#dc852f' if 'CRC' in group else '#349553'
            x,y=15+(i%cols)*12,45+(i//cols)*12
            c.create_rectangle(x,y,x+10,y+10,fill=color,outline='')


def launch(tx):
    App(tx).mainloop()
