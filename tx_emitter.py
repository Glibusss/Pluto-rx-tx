"""HackRF TX-emitter GUI."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hackrf_emitter import (EmitterConfig, HACKRF_SAMPLE_RATES, MODS,
                            discover_hackrf_devices, find_hackrf_transfer,
                            run_transmitter)


class EmitterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('HackRF — TX-излучатель')
        self.geometry('980x760')
        self.minsize(820, 680)
        self.events = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.worker = None
        self.discovery_worker = None
        self.controls = []
        self.closing = False
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.build()
        self.after(100, self.poll)
        self.after(250, lambda: self.start_discovery(include_devices=True, silent=True))

    def _entry(self, parent, label, variable, row, column, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=column * 2, sticky='w',
                                           padx=5, pady=5)
        widget = ttk.Entry(parent, textvariable=variable, width=width)
        widget.grid(row=row, column=column * 2 + 1, sticky='ew', padx=5, pady=5)
        self.controls.append((widget, 'normal'))
        return widget

    def build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill='both', expand=True)

        device = ttk.LabelFrame(main, text='HackRF и RF-параметры', padding=8)
        device.pack(fill='x')
        for column in range(7):
            device.columnconfigure(column, weight=1 if column % 2 else 0)
        self.executable = tk.StringVar(value='hackrf_transfer')
        self.serial = tk.StringVar()
        self.frequency = tk.StringVar(value='2400')
        self.sample_rate = tk.StringVar(value='8')
        self.bandwidth = tk.StringVar(value='1')
        self.gain = tk.StringVar(value='0')
        self.amplitude = tk.StringVar(value='35')
        self.rf_amp = tk.BooleanVar(value=False)
        ttk.Label(device, text='hackrf_transfer').grid(row=0, column=0, sticky='w', padx=5)
        self.executable_widget = ttk.Entry(device, textvariable=self.executable, width=26)
        self.executable_widget.grid(row=0, column=1, sticky='ew', padx=5)
        browse = ttk.Button(device, text='Обзор…', command=self.pick_executable)
        browse.grid(row=0, column=2, sticky='w', padx=5)
        self.tools_button = ttk.Button(
            device, text='Найти Tools',
            command=lambda: self.start_discovery(include_devices=False))
        self.tools_button.grid(row=0, column=3, sticky='w', padx=5)
        ttk.Label(device, text='HackRF / Serial').grid(row=0, column=4, sticky='w', padx=5)
        self.serial_widget = ttk.Combobox(device, textvariable=self.serial, state='normal', width=23)
        self.serial_widget.grid(row=0, column=5, sticky='ew', padx=5)
        self.device_button = ttk.Button(
            device, text='Найти HackRF',
            command=lambda: self.start_discovery(include_devices=True))
        self.device_button.grid(row=0, column=6, sticky='w', padx=5)
        self.controls += [(self.executable_widget, 'normal'), (browse, 'normal'),
                          (self.tools_button, 'normal'), (self.serial_widget, 'normal'),
                          (self.device_button, 'normal')]
        self._entry(device, 'Частота центра, МГц', self.frequency, 1, 0)
        ttk.Label(device, text='Sample rate, MS/s').grid(row=1, column=2, sticky='w', padx=5)
        rate = ttk.Combobox(device, textvariable=self.sample_rate,
                            values=tuple(str(v // 1_000_000) for v in HACKRF_SAMPLE_RATES),
                            state='readonly', width=10)
        rate.grid(row=1, column=3, sticky='ew', padx=5)
        self.controls.append((rate, 'readonly'))
        self._entry(device, 'Полоса шума / размах, МГц', self.bandwidth, 1, 2)
        self._entry(device, 'TX VGA gain, dB (0…47)', self.gain, 2, 0)
        self._entry(device, 'Цифровая амплитуда, %', self.amplitude, 2, 1)
        amp = ttk.Checkbutton(device, text='RF amp (~11 dB, не dBm)', variable=self.rf_amp)
        amp.grid(row=2, column=4, columnspan=2, sticky='w', padx=5)
        self.controls.append((amp, 'normal'))

        signal = ttk.LabelFrame(main, text='Формирование сигнала', padding=8)
        signal.pack(fill='x', pady=10)
        self.signal_mode = tk.StringVar(value='packet')
        self.modulation = tk.StringVar(value='BPSK')
        self.iq_path = tk.StringVar()
        self.iq_format = tk.StringVar(value='CS8 — signed int8 I/Q')
        self.tuning_mode = tk.StringVar(value='fixed')
        self.sweep_period = tk.StringVar(value='100')
        self.operation_mode = tk.StringVar(value='continuous')
        self.pulse_on = tk.StringVar(value='50')
        self.pulse_off = tk.StringVar(value='50')

        ttk.Label(signal, text='Спектр').grid(row=0, column=0, sticky='w', padx=5)
        packet = ttk.Radiobutton(signal, text='Кадры основного передатчика',
                                 variable=self.signal_mode, value='packet',
                                 command=self.update_modes)
        packet.grid(row=0, column=1, sticky='w', padx=5)
        noise = ttk.Radiobutton(signal, text='Сплошной спектр (полосовой шум)',
                                variable=self.signal_mode, value='noise',
                                command=self.update_modes)
        noise.grid(row=0, column=2, columnspan=2, sticky='w', padx=5)
        iq_file = ttk.Radiobutton(signal, text='IQ-файл', variable=self.signal_mode,
                                  value='file', command=self.update_modes)
        iq_file.grid(row=0, column=4, sticky='w', padx=5)
        self.controls += [(packet, 'normal'), (noise, 'normal'), (iq_file, 'normal')]
        ttk.Label(signal, text='Модуляция кадра').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.mod_widget = ttk.Combobox(signal, textvariable=self.modulation, values=MODS,
                                       state='readonly', width=18)
        self.mod_widget.grid(row=1, column=1, sticky='w', padx=5)
        self.controls.append((self.mod_widget, 'readonly'))

        ttk.Label(signal, text='Файл отсчётов').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.iq_path_widget = ttk.Entry(signal, textvariable=self.iq_path, width=42)
        self.iq_path_widget.grid(row=2, column=1, columnspan=2, sticky='ew', padx=5)
        self.iq_button = ttk.Button(signal, text='Выбрать…', command=self.pick_iq_file)
        self.iq_button.grid(row=2, column=3, sticky='w', padx=5)
        self.iq_format_widget = ttk.Combobox(
            signal, textvariable=self.iq_format,
            values=('CS8 — signed int8 I/Q', 'CF32 — complex64 little-endian'),
            state='readonly', width=29)
        self.iq_format_widget.grid(row=2, column=4, sticky='w', padx=5)
        self.controls += [(self.iq_path_widget, 'normal'), (self.iq_button, 'normal'),
                          (self.iq_format_widget, 'readonly')]

        ttk.Label(signal, text='Частота').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        fixed = ttk.Radiobutton(signal, text='Постоянная по центру', variable=self.tuning_mode,
                                value='fixed', command=self.update_modes)
        fixed.grid(row=3, column=1, sticky='w', padx=5)
        sweep = ttk.Radiobutton(signal, text='Скользящая в заданной полосе',
                                variable=self.tuning_mode, value='sweep',
                                command=self.update_modes)
        sweep.grid(row=3, column=2, sticky='w', padx=5)
        self.controls += [(fixed, 'normal'), (sweep, 'normal')]
        ttk.Label(signal, text='Период прохода, мс').grid(row=3, column=3, sticky='e', padx=5)
        self.sweep_widget = ttk.Entry(signal, textvariable=self.sweep_period, width=10)
        self.sweep_widget.grid(row=3, column=4, sticky='w', padx=5)
        self.controls.append((self.sweep_widget, 'normal'))

        ttk.Label(signal, text='Работа').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        continuous = ttk.Radiobutton(signal, text='Непрерывно', variable=self.operation_mode,
                                     value='continuous', command=self.update_modes)
        continuous.grid(row=4, column=1, sticky='w', padx=5)
        pulse = ttk.Radiobutton(signal, text='Импульсно', variable=self.operation_mode,
                                value='pulse', command=self.update_modes)
        pulse.grid(row=4, column=2, sticky='w', padx=5)
        self.controls += [(continuous, 'normal'), (pulse, 'normal')]
        pulse_times = ttk.Frame(signal)
        pulse_times.grid(row=4, column=3, columnspan=2, sticky='w')
        ttk.Label(pulse_times, text='ON, мс').pack(side='left')
        self.on_widget = ttk.Entry(pulse_times, textvariable=self.pulse_on, width=8)
        self.on_widget.pack(side='left', padx=(4, 10))
        ttk.Label(pulse_times, text='OFF, мс').pack(side='left')
        self.off_widget = ttk.Entry(pulse_times, textvariable=self.pulse_off, width=8)
        self.off_widget.pack(side='left', padx=4)
        self.controls += [(self.on_widget, 'normal'), (self.off_widget, 'normal')]
        ttk.Label(signal, text=(
            'В режиме «кадры» используется реальный формат modem.py. В режиме шума '
            'фиксированный сигнал заполняет полосу; при сканировании узкая шумовая полоса '
            'проходит по всей заданной полосе. В импульсном сканировании каждый импульс '
            'выполняет один полный проход; поле периода относится к непрерывному режиму. '
            'Для IQ-файла sample rate берётся из RF-параметров, а не из метаданных файла.'),
            wraplength=880, justify='left').grid(
                row=5, column=0, columnspan=5, sticky='w', padx=5, pady=(8, 0))

        run = ttk.Frame(main)
        run.pack(fill='x')
        self.start_button = ttk.Button(run, text='Start TX', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(run, text='Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        self.status = ttk.Label(run, text='Остановлено')
        self.status.pack(side='left', padx=8)

        log_frame = ttk.LabelFrame(main, text='Лог', padding=6)
        log_frame.pack(fill='both', expand=True, pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=14, state='disabled', wrap='word',
                           font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.write_log('Готово. Для работы требуются HackRF Tools и подключённый HackRF.')
        self.update_modes()

    def pick_executable(self):
        path = filedialog.askopenfilename(
            title='Укажите hackrf_transfer',
            filetypes=[('hackrf_transfer', 'hackrf_transfer.exe'), ('Все файлы', '*.*')])
        if path:
            self.executable.set(path)
            self.start_discovery(include_devices=True)

    def start_discovery(self, include_devices=True, silent=False):
        if ((self.worker is not None and self.worker.is_alive()) or
                (self.discovery_worker is not None and self.discovery_worker.is_alive())):
            return
        preferred = self.executable.get().strip()
        self.tools_button.configure(state='disabled')
        self.device_button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.status.configure(text='Поиск HackRF…' if include_devices else 'Поиск HackRF Tools…')

        def work():
            try:
                transfer = find_hackrf_transfer(preferred)
                if not transfer:
                    raise FileNotFoundError(
                        'hackrf_transfer не найден в PATH или стандартных каталогах установки.')
                self.emit('tools_found', transfer)
                if include_devices:
                    info_path, serials, output = discover_hackrf_devices(transfer)
                    self.emit('devices_found', (info_path, serials, output))
            except Exception as error:
                self.emit('discovery_error', (str(error), silent))
            finally:
                self.emit('discovery_done', None)

        self.discovery_worker = threading.Thread(
            target=work, name='HackRF discovery', daemon=True)
        self.discovery_worker.start()

    def pick_iq_file(self):
        path = filedialog.askopenfilename(
            title='Выберите файл IQ-отсчётов',
            filetypes=[('IQ CS8 / CF32', '*.cs8 *.iq *.cf32 *.c64'),
                       ('Все файлы', '*.*')])
        if path:
            self.iq_path.set(path)
            if path.lower().endswith(('.cf32', '.c64')):
                self.iq_format.set('CF32 — complex64 little-endian')
            self.signal_mode.set('file')
            self.update_modes()

    @staticmethod
    def _number(value):
        return float(value.strip().replace(',', '.'))

    def settings(self):
        gain = self._number(self.gain.get())
        if not gain.is_integer():
            raise ValueError('TX VGA gain задаётся целым числом dB')
        cfg = EmitterConfig(
            frequency_hz=int(self._number(self.frequency.get()) * 1e6),
            sample_rate=int(self._number(self.sample_rate.get()) * 1e6),
            bandwidth_hz=int(self._number(self.bandwidth.get()) * 1e6),
            txvga_gain=int(gain),
            rf_amp=self.rf_amp.get(),
            amplitude=self._number(self.amplitude.get()) / 100,
            signal_mode=self.signal_mode.get(), modulation=self.modulation.get(),
            tuning_mode=self.tuning_mode.get(), operation_mode=self.operation_mode.get(),
            sweep_period_ms=self._number(self.sweep_period.get()),
            pulse_on_ms=self._number(self.pulse_on.get()),
            pulse_off_ms=self._number(self.pulse_off.get()),
            iq_path=self.iq_path.get().strip(),
            iq_format='cf32' if self.iq_format.get().startswith('CF32') else 'cs8',
            serial=self.serial.get().strip(), executable=self.executable.get().strip())
        cfg.validate()
        return cfg

    def update_modes(self):
        running = self.worker is not None and self.worker.is_alive()
        if running:
            return
        self.mod_widget.configure(state='readonly' if self.signal_mode.get() == 'packet' else 'disabled')
        file_state = self.signal_mode.get() == 'file'
        self.iq_path_widget.configure(state='normal' if file_state else 'disabled')
        self.iq_button.configure(state='normal' if file_state else 'disabled')
        self.iq_format_widget.configure(state='readonly' if file_state else 'disabled')
        self.sweep_widget.configure(state='normal' if self.tuning_mode.get() == 'sweep' else 'disabled')
        pulse_state = 'normal' if self.operation_mode.get() == 'pulse' else 'disabled'
        self.on_widget.configure(state=pulse_state)
        self.off_widget.configure(state=pulse_state)

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        if self.discovery_worker is not None and self.discovery_worker.is_alive():
            self.write_log('Дождитесь завершения поиска HackRF.')
            return
        try:
            cfg = self.settings()
        except Exception as error:
            messagebox.showerror('Параметры', str(error))
            return
        self.stop_event.clear()
        for widget, _state in self.controls:
            widget.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.configure(text='Подготовка…')
        self.write_log(
            f'Центр {cfg.frequency_hz / 1e6:g} МГц; полоса {cfg.bandwidth_hz / 1e6:g} МГц; '
            f'Fs {cfg.sample_rate / 1e6:g} MS/s; TX VGA {cfg.txvga_gain} dB; '
            f'RF amp {"ON" if cfg.rf_amp else "OFF"}. Это не значение dBm.')

        def work():
            try:
                run_transmitter(cfg, self.stop_event, self.emit)
            except Exception:
                self.emit('error', traceback.format_exc())
            finally:
                self.emit('done', None)

        self.worker = threading.Thread(target=work, name='HackRF TX-emitter', daemon=True)
        self.worker.start()

    def emit(self, kind, value):
        self.events.put((kind, value))

    def stop(self):
        self.stop_event.set()
        self.stop_button.configure(state='disabled')
        self.status.configure(text='Остановка…')

    def close(self):
        self.closing = True
        if self.worker is not None and self.worker.is_alive():
            self.stop()
        else:
            self.destroy()

    def write_log(self, value):
        self.log.configure(state='normal')
        self.log.insert('end', time.strftime('%H:%M:%S') + ' ' + value + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.write_log(value)
            elif kind == 'status':
                self.status.configure(text=value)
            elif kind == 'error':
                self.write_log(value)
                messagebox.showerror('HackRF TX', value.splitlines()[-1] if value else 'Ошибка')
            elif kind == 'tools_found':
                self.executable.set(value)
                self.write_log('Найден HackRF Tools: ' + value)
            elif kind == 'devices_found':
                info_path, serials, _output = value
                self.serial_widget.configure(values=serials)
                current = self.serial.get().strip()
                if serials and not current:
                    self.serial.set(serials[0])
                if serials:
                    self.write_log(f'hackrf_info: найдено устройств {len(serials)}; '
                                   + ', '.join(serials))
                else:
                    self.write_log('hackrf_info не обнаружил подключённых устройств.')
                self.write_log('Использован: ' + info_path)
            elif kind == 'discovery_error':
                error, silent = value
                self.write_log('Автопоиск: ' + error)
                if not silent:
                    messagebox.showerror('Автопоиск HackRF', error)
            elif kind == 'discovery_done':
                self.tools_button.configure(state='normal')
                self.device_button.configure(state='normal')
                if self.worker is None or not self.worker.is_alive():
                    self.start_button.configure(state='normal')
                    self.status.configure(text='Остановлено')
            elif kind == 'done':
                for widget, state in self.controls:
                    widget.configure(state=state)
                self.start_button.configure(state='normal')
                self.stop_button.configure(state='disabled')
                self.status.configure(text='Остановлено')
                self.update_modes()
                if self.closing:
                    self.destroy()
                    return
        self.after(100, self.poll)


def launch():
    EmitterApp().mainloop()


if __name__ == '__main__':
    launch()
