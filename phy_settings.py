"""One editable PHY profile shared by Pluto TX/RX and the HackRF emitter."""
import tkinter as tk
from tkinter import ttk
from modem import Config
from modulations import MODS, SPS, constellation


class PhySettings(ttk.LabelFrame):
    def __init__(self, parent, controls):
        super().__init__(parent, text='Профиль сигнала — одинаковый на TX и RX', padding=6)
        self.controls = controls
        defaults = Config()
        names = ('mod', 'preamble_symbols', 'preamble_seed', 'cp_symbols', 'pilot_symbols',
                 'pilot_seed', 'data_symbols', 'payload_bytes', 'phase_deg', 'ook_low')
        for name in names:
            setattr(self, name, tk.StringVar(value=str(getattr(defaults, name))))
        self.fsk_deviation_mhz = tk.StringVar(value=f'{defaults.fsk_deviation_hz/1_000_000:g}')
        self.preamble = tk.BooleanVar(value=True)
        self.cp = tk.BooleanVar(value=False)
        ttk.Label(self, text='Модуляция').grid(row=0, column=0, sticky='w', padx=4)
        widget = ttk.Combobox(self, textvariable=self.mod, values=MODS, state='readonly', width=16)
        widget.grid(row=0, column=1, sticky='w', padx=4)
        self.mod_widget = widget
        controls.append((widget, 'readonly'))
        self.description = ttk.Label(self)
        self.description.grid(row=0, column=2, columnspan=4, sticky='w', padx=8)
        for col, (text, var) in enumerate((('Преамбула', self.preamble), ('CP / SC-FDE', self.cp))):
            widget = ttk.Checkbutton(self, text=text, variable=var)
            widget.grid(row=1, column=col*2, sticky='w', padx=4)
            controls.append((widget, 'normal'))
        fields = [('Символов преамбулы', 'preamble_symbols'), ('Seed преамбулы', 'preamble_seed'),
                  ('Символов CP', 'cp_symbols'), ('Пилотов в блоке', 'pilot_symbols'),
                  ('Seed пилотов', 'pilot_seed'), ('Символов данных в блоке', 'data_symbols'),
                  ('Данных в пакете, байт', 'payload_bytes')]
        for index, (text, name) in enumerate(fields):
            row, pair = 2+index//3, index%3
            self.entry(self, text, getattr(self, name), row, pair)
        self.mode_frame = ttk.Frame(self)
        self.mode_frame.grid(row=5, column=0, columnspan=6, sticky='w', pady=(5,0))
        self.mode_widgets = []
        self.mod.trace_add('write', self.update_modulation)
        self.update_modulation()
        ttk.Label(self, text='Все части кадра используют выбранную модуляцию. CP помогает при отражениях; '
                  'для FSK служит защитным интервалом.', wraplength=1040).grid(
                      row=6, column=0, columnspan=6, sticky='w', padx=4, pady=(5,0))

    def entry(self, parent, label, variable, row, pair):
        ttk.Label(parent, text=label).grid(row=row, column=pair*2, sticky='w', padx=4, pady=3)
        widget = ttk.Entry(parent, textvariable=variable, width=10)
        widget.grid(row=row, column=pair*2+1, sticky='w', padx=4, pady=3)
        self.controls.append((widget, 'normal'))
        return widget

    def update_modulation(self, *_):
        for widget in self.mode_widgets:
            self.controls[:] = [(w,s) for w,s in self.controls if w is not widget]
        self.mode_widgets.clear()
        for child in self.mode_frame.winfo_children():
            child.destroy()
        mod = self.mod.get()
        pts, k = constellation(mod)
        self.description.configure(text=f'{len(pts)} состояний · {k} бит/символ · Rs = Fs/{SPS}')
        if mod == 'BPSK':
            ttk.Label(self.mode_frame, text='BPSK: дополнительных настроек нет').grid(row=0, column=0, padx=4)
            return
        if mod == '2-FSK':
            label, variable = 'Девиация Δf, МГц (тоны f₀ ± Δf)', self.fsk_deviation_mhz
        elif mod == 'ASK / OOK':
            label, variable = 'Нижний / верхний уровень (0 = OOK; до 0.9)', self.ook_low
        else:
            label, variable = 'Поворот созвездия, ° (−180…180)', self.phase_deg
        self.mode_widgets.append(self.entry(self.mode_frame, label, variable, 0, 0))

    def config(self, sample_rate=1_000_000, cfo_range=40_000):
        integer_names = ('preamble_symbols', 'preamble_seed', 'cp_symbols', 'pilot_symbols',
                         'pilot_seed', 'data_symbols', 'payload_bytes')
        values = {name: int(getattr(self, name).get()) for name in integer_names}
        mod = self.mod.get()
        parameter = {'2-FSK': 'fsk_deviation_hz', 'ASK / OOK': 'ook_low'}.get(mod, 'phase_deg')
        if mod == '2-FSK':
            values[parameter] = float(self.fsk_deviation_mhz.get().replace(',', '.')) * 1_000_000
        elif mod != 'BPSK':
            values[parameter] = float(getattr(self, parameter).get().replace(',', '.'))
        cfg = Config(mod=self.mod.get(), preamble=self.preamble.get(), cp=self.cp.get(),
                     sample_rate=sample_rate, cfo_range=cfo_range, **values)
        cfg.validate()
        return cfg

    def set_enabled(self, enabled):
        def visit(parent):
            for widget in parent.winfo_children():
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state='readonly' if enabled else 'disabled')
                elif isinstance(widget, (ttk.Entry, ttk.Checkbutton)):
                    widget.configure(state='normal' if enabled else 'disabled')
                visit(widget)
        visit(self)
