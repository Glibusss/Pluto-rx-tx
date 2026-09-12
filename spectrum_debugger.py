"""Live PlutoSDR spectrum viewer independent from the packet decoder."""
from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


ADC_FULL_SCALE = 2048.0  # AD936x signed 12-bit converter values returned by libiio.


def calculate_spectrum(iq, sample_rate, center_frequency, full_scale=ADC_FULL_SCALE):
    """Return absolute frequency bins and linear power relative to ADC full scale."""
    samples = np.asarray(iq, np.complex64).ravel()
    if len(samples) < 16:
        raise ValueError('Для FFT нужно не менее 16 комплексных отсчётов')
    if sample_rate <= 0 or full_scale <= 0:
        raise ValueError('Sample rate и full scale должны быть положительными')
    window = np.hanning(len(samples)).astype(np.float32)
    scale = full_scale*float(window.sum())
    transformed = np.fft.fftshift(np.fft.fft(samples*window))/scale
    power = np.maximum(abs(transformed)**2,1e-20)
    frequency = center_frequency+np.fft.fftshift(np.fft.fftfreq(len(samples),1/sample_rate))
    return frequency,power


class SpectrumDebugger(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PlutoSDR — отладчик спектра')
        self.geometry('1100x700')
        self.minsize(760,500)
        self.events=queue.SimpleQueue()
        self.latest_lock=threading.Lock()
        self.latest=None
        self.stop_event=threading.Event()
        self.worker=None
        self.closing=False
        self.controls=[]
        self.trace=None
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.build()
        self.after(100,self.poll)

    def build(self):
        main=ttk.Frame(self,padding=10)
        main.pack(fill='both',expand=True)
        settings=ttk.LabelFrame(main,text='Параметры PlutoSDR',padding=8)
        settings.pack(fill='x')
        self.uri=tk.StringVar(value='ip:192.168.2.1')
        self.frequency=tk.StringVar(value='2400')
        self.sample_rate=tk.StringVar(value='1')
        self.gain=tk.StringVar(value='20')
        self.averaging=tk.StringVar(value='5')
        fields=[('URI',self.uri,22),('Центр, МГц',self.frequency,12),
                ('RX gain, dB',self.gain,10),('Усреднение FFT',self.averaging,10)]
        for column,(label,var,width) in enumerate(fields):
            ttk.Label(settings,text=label).grid(row=0,column=column*2,sticky='w',padx=4)
            widget=ttk.Entry(settings,textvariable=var,width=width)
            widget.grid(row=0,column=column*2+1,sticky='ew',padx=4)
            self.controls.append((widget,'normal'))
        ttk.Label(settings,text='Sample rate, MS/s').grid(row=1,column=0,sticky='w',padx=4,pady=(8,0))
        widget=ttk.Combobox(settings,textvariable=self.sample_rate,values=('1','2'),state='readonly',width=10)
        widget.grid(row=1,column=1,sticky='w',padx=4,pady=(8,0))
        self.controls.append((widget,'readonly'))
        self.ignore_dc=tk.BooleanVar(value=True)
        widget=ttk.Checkbutton(settings,text='Не учитывать центральный DC-пик при поиске максимума',variable=self.ignore_dc)
        widget.grid(row=1,column=2,columnspan=6,sticky='w',padx=4,pady=(8,0))

        actions=ttk.Frame(main)
        actions.pack(fill='x',pady=8)
        self.start_button=ttk.Button(actions,text='Start spectrum',command=self.start)
        self.start_button.pack(side='left')
        self.stop_button=ttk.Button(actions,text='Stop',command=self.stop,state='disabled')
        self.stop_button.pack(side='left',padx=8)
        self.status=ttk.Label(actions,text='Остановлено')
        self.status.pack(side='left',padx=8)

        self.measurements=ttk.Label(main,text='Пик: —    Смещение: —    Уровень: —    Фон: —',
                                    font=('Segoe UI',11))
        self.measurements.pack(fill='x',pady=(0,8))
        self.canvas=tk.Canvas(main,bg='#10151c',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda event:self.draw())
        ttk.Label(main,text=('Вертикальная линия в центре — заданная частота LO. Полоса обзора равна sample rate. '
                             'Этот режим показывает сырой эфир и не запускает пакетный декодер.'),
                  wraplength=1060).pack(fill='x',pady=(8,0))

    def read_settings(self):
        uri=self.uri.get().strip()
        if not uri:
            raise ValueError('Введите URI Pluto')
        center=int(float(self.frequency.get().replace(',','.'))*1e6)
        if not 326_000_000<=center<=3_800_000_000:
            raise ValueError('Центральная частота: 326…3800 МГц')
        rate=int(float(self.sample_rate.get())*1e6)
        if rate not in (1_000_000,2_000_000):
            raise ValueError('Sample rate: 1 или 2 MS/s')
        gain=float(self.gain.get().replace(',','.'))
        if not -3<=gain<=71:
            raise ValueError('RX gain: -3…71 dB')
        averaging=int(self.averaging.get())
        if not 1<=averaging<=100:
            raise ValueError('Усреднение FFT: 1…100')
        return dict(uri=uri,center=center,rate=rate,gain=gain,averaging=averaging)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            settings=self.read_settings()
        except Exception as error:
            messagebox.showerror('Параметры',str(error))
            return
        self.stop_event.clear()
        self.trace=None
        for widget,_ in self.controls:
            widget.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.configure(text='Подключение…')
        self.worker=threading.Thread(target=self.capture,args=(settings,),name='Pluto spectrum',daemon=True)
        self.worker.start()

    def capture(self,settings):
        sdr=None
        try:
            try:
                import adi
            except (ImportError,OSError) as error:
                raise RuntimeError('Не удалось загрузить pyadi-iio/libiio: '+str(error)) from error
            sdr=adi.Pluto(uri=settings['uri'])
            sdr._ctx.set_timeout(2000)
            sdr.rx_enabled_channels=[0]
            sdr.sample_rate=settings['rate']
            sdr.rx_lo=settings['center']
            sdr.rx_rf_bandwidth=settings['rate']
            sdr.gain_control_mode_chan0='manual'
            sdr.rx_hardwaregain_chan0=settings['gain']
            sdr.rx_buffer_size=32768
            sdr._rxadc.set_kernel_buffers_count(4)
            actual_center=int(sdr.rx_lo)
            actual_rate=int(sdr.sample_rate)
            self.events.put(('connected',(actual_center,actual_rate)))
            average=None
            count=settings['averaging']
            while not self.stop_event.is_set():
                raw=np.asarray(sdr.rx(),np.complex64)
                frequency,power=calculate_spectrum(raw,actual_rate,actual_center)
                average=power if average is None else average+(power-average)/count
                db=10*np.log10(np.maximum(average,1e-20))
                clipping=float(np.mean((abs(raw.real)>=2040)|(abs(raw.imag)>=2040)))
                with self.latest_lock:
                    self.latest=(frequency,db,actual_center,actual_rate,clipping)
        except Exception:
            self.events.put(('error',traceback.format_exc()))
        finally:
            if sdr is not None:
                try:sdr.rx_destroy_buffer()
                except Exception:pass
            self.events.put(('done',None))

    def stop(self):
        self.stop_event.set()
        self.stop_button.configure(state='disabled')
        self.status.configure(text='Остановка…')

    def close(self):
        self.closing=True
        if self.worker and self.worker.is_alive():
            self.stop()
        else:
            self.destroy()

    def poll(self):
        with self.latest_lock:
            trace=self.latest
            self.latest=None
        if trace is not None:
            self.trace=trace
            self.update_measurements()
            self.draw()
        while True:
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind=='connected':
                self.status.configure(text=f'Подключено · LO {value[0]/1e6:.6f} МГц · Fs {value[1]/1e6:g} MS/s')
            elif kind=='error':
                self.status.configure(text='Ошибка')
                messagebox.showerror('Спектр PlutoSDR',value)
            elif kind=='done':
                for widget,state in self.controls:
                    widget.configure(state=state)
                self.start_button.configure(state='normal')
                self.stop_button.configure(state='disabled')
                if self.status.cget('text')!='Ошибка':
                    self.status.configure(text='Остановлено')
                if self.closing:
                    self.destroy()
                    return
        self.after(100,self.poll)

    def peak_index(self):
        frequency,db,center,rate,_=self.trace
        mask=np.ones(len(db),bool)
        if self.ignore_dc.get():
            mask &= abs(frequency-center)>=max(3000,3*rate/len(db))
        return int(np.argmax(np.where(mask,db,-np.inf))),mask

    def update_measurements(self):
        frequency,db,center,_,clipping=self.trace
        index,mask=self.peak_index()
        peak=frequency[index]
        noise=float(np.median(db[mask]))
        clip_text=f'    Клиппинг: {clipping:.2%}' if clipping else ''
        self.measurements.configure(text=(f'Пик: {peak/1e6:.6f} МГц    '
            f'Смещение: {(peak-center)/1000:+.2f} кГц    '
            f'Уровень: {db[index]:.1f} dBFS    Фон: {noise:.1f} dBFS{clip_text}'))

    def draw(self):
        canvas=self.canvas
        canvas.delete('all')
        width=max(canvas.winfo_width(),300)
        height=max(canvas.winfo_height(),220)
        left,right,top,bottom=70,20,20,45
        plot_width=max(1,width-left-right)
        plot_height=max(1,height-top-bottom)
        ymin,ymax=-120,5
        for level in range(-120,1,20):
            y=top+(ymax-level)/(ymax-ymin)*plot_height
            canvas.create_line(left,y,width-right,y,fill='#283341')
            canvas.create_text(left-8,y,text=str(level),fill='#aeb9c5',anchor='e')
        canvas.create_text(8,10,text='dBFS',fill='#aeb9c5',anchor='nw')
        if self.trace is None:
            canvas.create_text(width/2,height/2,text='Нажмите Start spectrum',fill='#aeb9c5',font=('Segoe UI',14))
            return
        frequency,db,center,rate,_=self.trace
        bins_per_pixel=max(1,int(np.ceil(len(db)/plot_width)))
        usable=(len(db)//bins_per_pixel)*bins_per_pixel
        reduced=db[:usable].reshape(-1,bins_per_pixel).max(axis=1)
        reduced_frequency=frequency[:usable].reshape(-1,bins_per_pixel).mean(axis=1)
        xmin,xmax=center-rate/2,center+rate/2
        x=left+(reduced_frequency-xmin)/(xmax-xmin)*plot_width
        y=top+(ymax-np.clip(reduced,ymin,ymax))/(ymax-ymin)*plot_height
        coordinates=np.column_stack((x,y)).ravel().tolist()
        if len(coordinates)>=4:
            canvas.create_line(*coordinates,fill='#45c4ff',width=1)
        center_x=left+plot_width/2
        canvas.create_line(center_x,top,center_x,height-bottom,fill='#ffb84d',dash=(5,4))
        for index in range(5):
            value=xmin+(xmax-xmin)*index/4
            px=left+plot_width*index/4
            canvas.create_line(px,height-bottom,px,height-bottom+5,fill='#aeb9c5')
            canvas.create_text(px,height-bottom+8,text=f'{value/1e6:.3f}',fill='#aeb9c5',anchor='n')
        canvas.create_text(width/2,height-5,text='Частота, МГц',fill='#aeb9c5',anchor='s')


if __name__=='__main__':
    SpectrumDebugger().mainloop()
