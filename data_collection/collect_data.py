#!/usr/bin/env python3
# -*-coding:utf-8-*-
# SPDX-FileCopyrightText: 2021-2025 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0

import sys
import csv
import json
import argparse
import numpy as np
import serial
from os import path, makedirs
from io import StringIO
from datetime import datetime

from PyQt5.Qt import *
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QThread
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ScatterPlotItem

# -----------------------------
# Existing constants / buffers
# -----------------------------
CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_len = 0

CSI_DATA_INDEX = 200  # ring buffer for plotting (last N frames)

# NOTE: This is "max" for plotting; real used subcarriers depends on csi_data_len
CSI_DATA_COLUMNS = 490

DATA_COLUMNS_NAMES_C5C6 = ['type', 'id', 'mac', 'rssi', 'rate','noise_floor','fft_gain','agc_gain',
                           'channel', 'local_timestamp',  'sig_len', 'rx_state', 'len', 'first_word', 'data']

DATA_COLUMNS_NAMES = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth', 'smoothing',
                      'not_sounding', 'aggregation', 'stbc', 'fec_coding','sgi', 'noise_floor', 'ampdu_cnt',
                      'channel', 'secondary_channel', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len',
                      'first_word', 'data']

csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)

# -----------------------------
# Dataset collection config
# -----------------------------
OUTPUT_DIR = "../csi_dataset"
LABELS = {
    0: "No Human",
    1: "Human (Static)",
    2: "Human (Movement)",
}

# We will store amplitude for ONLY the active subcarriers (csi_data_len//2) each frame.
# Each row: timestamp,label,rssi,noise_floor,channel,fft_gain,agc_gain, amp_0..amp_N-1
# (N depends on your current CSI length)
# -----------------------------

def generate_subcarrier_colors(red_range, green_range, yellow_range, total_num, interval=1):
    colors = []
    for i in range(total_num):
        if red_range and red_range[0] <= i <= red_range[1] and red_range[1] != red_range[0]:
            intensity = int(255 * (i - red_range[0]) / (red_range[1] - red_range[0]))
            colors.append((intensity, 0, 0))
        elif green_range and green_range[0] <= i <= green_range[1] and green_range[1] != green_range[0]:
            intensity = int(255 * (i - green_range[0]) / (green_range[1] - green_range[0]))
            colors.append((0, intensity, 0))
        elif yellow_range and yellow_range[0] <= i <= yellow_range[1] and yellow_range[1] != yellow_range[0]:
            intensity = int(255 * (i - yellow_range[0]) / (yellow_range[1] - yellow_range[0]))
            colors.append((0, intensity, intensity))
        else:
            colors.append((200, 200, 200))
    return colors


class DatasetRecorder:
    """
    Collect labeled samples based on the SAME parsed frames that drive the plot.

    - start()/stop(): toggles recording
    - set_label(): set 0/1/2
    - add_sample(): called from serial thread when a frame is parsed
    - save_csv(): dumps to CSV
    """
    def __init__(self):
        self.is_recording = False
        self.current_label = 0
        self.samples = []
        self.active_amp_len = None  # set when first frame arrives

    def set_label(self, label: int):
        if label in LABELS:
            self.current_label = label

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        return self.is_recording

    def add_sample(self, *, local_timestamp: int, rssi: int, noise_floor: int, channel: int,
                   fft_gain: int, agc_gain: int, amps: np.ndarray):
        if not self.is_recording:
            return

        if self.active_amp_len is None:
            self.active_amp_len = int(len(amps))

        # If length changes mid-run, we can either:
        # - truncate/pad to first length OR
        # - start a new file
        # Here we pad/truncate to the first observed length for consistency.
        n = self.active_amp_len
        if len(amps) < n:
            amps_fixed = np.pad(amps, (0, n - len(amps)), constant_values=0.0)
        else:
            amps_fixed = amps[:n]

        row = [
            int(local_timestamp),
            int(self.current_label),
            int(rssi),
            int(noise_floor),
            int(channel),
            int(fft_gain),
            int(agc_gain),
        ] + [float(x) for x in amps_fixed.tolist()]

        self.samples.append(row)

    def save_csv(self, filename: str = None):
        if not self.samples:
            return None, "No data to save."

        makedirs(OUTPUT_DIR, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"csi_labeled_{ts}.csv"

        out_path = path.join(OUTPUT_DIR, filename)

        amp_len = self.active_amp_len if self.active_amp_len is not None else (len(self.samples[0]) - 7)
        header = ["timestamp", "label", "rssi", "noise_floor", "channel", "fft_gain", "agc_gain"] + \
                 [f"amp_{i}" for i in range(amp_len)]

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.samples)

        saved_count = len(self.samples)
        self.samples = []  # clear after save
        return out_path, f"Saved {saved_count} samples to {out_path}"


def csi_data_read_parse(port: str, csv_writer, log_file_fd,
                        callback_colors=None,
                        callback_sample=None):
    """
    Read serial, parse CSI, update global plot buffers, emit:
      - callback_colors(colors) once after first good frame
      - callback_sample(dict) on every good frame (for dataset collection)
    """
    ser = serial.Serial(port=port, baudrate=921600, bytesize=8, parity='N', stopbits=1, timeout=1)

    if ser.isOpen():
        print('open success')
    else:
        print('open failed')
        return

    first_good = True

    while True:
        strings = str(ser.readline())
        if not strings:
            continue

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        csv_reader = csv.reader(StringIO(strings))
        try:
            csi_data = next(csv_reader)
        except Exception:
            log_file_fd.write("csv parse failed\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # csi_data_len is stored near the end in esp-csi format; in this script it is [-3]
        try:
            csi_data_len = int(csi_data[-3])
        except Exception:
            log_file_fd.write("bad csi_data_len\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
            log_file_fd.write('element number is not equal\n')
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            log_file_fd.write('data is incomplete\n')
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        if csi_data_len != len(csi_raw_data):
            log_file_fd.write('csi_data_len is not equal\n')
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Depending on board/log format, indices differ. Your current script assumes:
        # fft_gain = csi_data[6], agc_gain = csi_data[7]
        # noise_floor = csi_data[5] OR csi_data[14] depending on format.
        # We'll keep your original assumptions for C5/C6 style.
        try:
            # Try C5/C6 layout first (has fft_gain/agc_gain at [6],[7])
            fft_gain = int(csi_data[6])
            agc_gain = int(csi_data[7])
            noise_floor = int(csi_data[5])
            channel = int(csi_data[8])
            local_timestamp = int(csi_data[9])
            rssi = int(csi_data[3])
        except Exception:
            # Fallback to older layout (best-effort)
            try:
                rssi = int(csi_data[3])
                noise_floor = int(csi_data[14])
                channel = int(csi_data[16])
                local_timestamp = int(csi_data[18])
                fft_gain = 0
                agc_gain = 0
            except Exception:
                log_file_fd.write("metadata parse failed\n")
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue

        csv_writer.writerow(csi_data)

        # Shift plot buffers
        csi_data_complex[:-1] = csi_data_complex[1:]
        agc_gain_data[:-1] = agc_gain_data[1:]
        fft_gain_data[:-1] = fft_gain_data[1:]
        agc_gain_data[-1] = agc_gain
        fft_gain_data[-1] = fft_gain

        # Determine active subcarriers count (I/Q pairs)
        active_sc = csi_data_len // 2
        if active_sc > CSI_DATA_COLUMNS:
            active_sc = CSI_DATA_COLUMNS

        # Fill latest row with complex values
        for i in range(active_sc):
            # Your original mapping:
            # complex(real=csi_raw_data[i*2+1], imag=csi_raw_data[i*2])
            csi_data_complex[-1][i] = complex(csi_raw_data[i * 2 + 1], csi_raw_data[i * 2])

        # zero remaining columns (so old data doesn't linger in plot)
        if active_sc < CSI_DATA_COLUMNS:
            csi_data_complex[-1][active_sc:] = 0 + 0j

        # Emit colors once (based on first frame)
        if first_good and callback_colors is not None:
            first_good = False
            raw_len = len(csi_raw_data)
            # Reuse your mapping logic
            if csi_data_len == 490:
                colors = generate_subcarrier_colors((0, 61), (62, 122), (123, 245), raw_len)
            elif csi_data_len == 234:
                colors = generate_subcarrier_colors((0, 28), (29, 56), (60, 116), raw_len)
            else:
                colors = generate_subcarrier_colors((0, raw_len // 2), (raw_len // 2 + 1, raw_len - 1), None, raw_len)

            callback_colors(colors)

        # Emit sample for dataset recorder
        if callback_sample is not None:
            amps = np.abs(csi_data_complex[-1, :active_sc]).astype(np.float32)
            callback_sample({
                "local_timestamp": local_timestamp,
                "rssi": rssi,
                "noise_floor": noise_floor,
                "channel": channel,
                "fft_gain": fft_gain,
                "agc_gain": agc_gain,
                "amps": amps,
            })

    ser.close()


class SubThread(QThread):
    data_ready = pyqtSignal(object)   # colors
    sample_ready = pyqtSignal(object) # per-frame metadata+amps

    def __init__(self, serial_port, save_file_name, log_file_name):
        super().__init__()
        self.serial_port = serial_port

        save_file_fd = open(save_file_name, 'w')
        self.log_file_fd = open(log_file_name, 'w')
        self.csv_writer = csv.writer(save_file_fd)

        # keep original CSV header
        self.csv_writer.writerow(DATA_COLUMNS_NAMES)

    def run(self):
        csi_data_read_parse(
            self.serial_port,
            self.csv_writer,
            self.log_file_fd,
            callback_colors=self.data_ready.emit,
            callback_sample=self.sample_ready.emit,
        )

    def __del__(self):
        self.wait()
        self.log_file_fd.close()


class csi_data_graphical_window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1280, 980)
        self.setWindowTitle("ESP CSI Viewer + Labeled Dataset Collector")

        # recorder
        self.recorder = DatasetRecorder()

        # ---------------- UI: status + buttons ----------------
        self.status = QLabel(self)
        self.status.setGeometry(QtCore.QRect(10, 910, 1260, 30))
        self.status.setText(self._status_text())

        btn_y = 940
        self.btn_label0 = QPushButton("Label 0: No Human", self)
        self.btn_label0.setGeometry(10, btn_y, 180, 30)
        self.btn_label0.clicked.connect(lambda: self.set_label(0))

        self.btn_label1 = QPushButton("Label 1: Static", self)
        self.btn_label1.setGeometry(200, btn_y, 160, 30)
        self.btn_label1.clicked.connect(lambda: self.set_label(1))

        self.btn_label2 = QPushButton("Label 2: Movement", self)
        self.btn_label2.setGeometry(370, btn_y, 180, 30)
        self.btn_label2.clicked.connect(lambda: self.set_label(2))

        self.btn_rec = QPushButton("Start/Stop Recording (R)", self)
        self.btn_rec.setGeometry(560, btn_y, 220, 30)
        self.btn_rec.clicked.connect(self.toggle_recording)

        self.btn_save = QPushButton("Save CSV (S)", self)
        self.btn_save.setGeometry(790, btn_y, 140, 30)
        self.btn_save.clicked.connect(self.save_data)

        self.btn_clear = QPushButton("Clear Buffer", self)
        self.btn_clear.setGeometry(940, btn_y, 120, 30)
        self.btn_clear.clicked.connect(self.clear_data)

        self.hint = QLabel(self)
        self.hint.setGeometry(QtCore.QRect(1070, btn_y, 200, 30))
        self.hint.setText("Keys: 0/1/2, R, S")

        # ---------------- Plots ----------------
        self.plotWidget_ted = PlotWidget(self)
        self.plotWidget_ted.setGeometry(QtCore.QRect(0, 0, 640, 300))
        self.plotWidget_ted.setYRange(-2*np.pi, 2*np.pi)
        self.plotWidget_ted.addLegend()
        self.plotWidget_ted.setTitle('Phase Data - Last Frame')
        self.plotWidget_ted.setLabel('left', 'Phase (rad)')
        self.plotWidget_ted.setLabel('bottom', 'Subcarrier Index')

        self.csi_amplitude_array = np.abs(csi_data_complex)
        self.csi_phase_array = np.angle(csi_data_complex)
        self.curve = self.plotWidget_ted.plot([], name='CSI Row Data', pen='r')

        self.plotWidget_multi_data = PlotWidget(self)
        self.plotWidget_multi_data.setGeometry(QtCore.QRect(0, 300, 1280, 300))
        self.plotWidget_multi_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_multi_data.addLegend()
        self.plotWidget_multi_data.setTitle('Subcarrier Amplitude Data')
        self.plotWidget_multi_data.setLabel('left', 'Amplitude')
        self.plotWidget_multi_data.setLabel('bottom', 'Time (Cumulative Packet Count)')

        self.curve_list = []
        agc_curve = self.plotWidget_multi_data.plot(agc_gain_data, name='AGC Gain', pen=[255,255,0])
        fft_curve = self.plotWidget_multi_data.plot(fft_gain_data, name='FFT Gain', pen=[255,255,0])
        self.curve_list.append(agc_curve)
        self.curve_list.append(fft_curve)

        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_multi_data.plot(self.csi_amplitude_array[:, i], name=str(i), pen=(255, 255, 255))
            self.curve_list.append(curve)

        self.plotWidget_phase_data = PlotWidget(self)
        self.plotWidget_phase_data.setGeometry(QtCore.QRect(0, 600, 1280, 300))
        self.plotWidget_phase_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_phase_data.addLegend()
        self.plotWidget_phase_data.setTitle('Subcarrier Phase Data')
        self.plotWidget_phase_data.setLabel('left', 'Phase (rad)')
        self.plotWidget_phase_data.setLabel('bottom', 'Time (Cumulative Packet Count)')

        self.curve_phase_list = []
        for i in range(CSI_DATA_COLUMNS):
            phase_curve = self.plotWidget_phase_data.plot(np.angle(self.csi_amplitude_array[:, i]), name=str(i), pen=(255, 255, 255))
            self.curve_phase_list.append(phase_curve)

        self.plotWidget_iq = PlotWidget(self)
        self.plotWidget_iq.setGeometry(QtCore.QRect(640, 0, 640, 300))
        self.plotWidget_iq.setLabel('left', 'Q (Imag)')
        self.plotWidget_iq.setLabel('bottom', 'I (Real)')
        self.plotWidget_iq.setTitle('IQ Plot - Last Frame')
        self.plotWidget_iq.getViewBox().setRange(QtCore.QRectF(-30, -30, 60, 60))
        self.plotWidget_iq.getViewBox().setAspectLocked(True)
        self.iq_scatter = ScatterPlotItem(size=6)
        self.plotWidget_iq.addItem(self.iq_scatter)

        self.iq_colors = []
        self.deta_len = 0

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    # --------- dataset controls ----------
    def _status_text(self):
        return (f"Label: {self.recorder.current_label} ({LABELS[self.recorder.current_label]}) | "
                f"Recording: {'ON' if self.recorder.is_recording else 'OFF'} | "
                f"Buffered samples: {len(self.recorder.samples)} | Output: ./{OUTPUT_DIR}")

    def set_label(self, label: int):
        self.recorder.set_label(label)
        self.status.setText(self._status_text())

    def toggle_recording(self):
        self.recorder.toggle_recording()
        self.status.setText(self._status_text())

    def clear_data(self):
        self.recorder.samples = []
        self.status.setText(self._status_text())

    def save_data(self):
        out_path, msg = self.recorder.save_csv()
        self.status.setText(self._status_text())
        if out_path:
            QMessageBox.information(self, "Saved", msg)
        else:
            QMessageBox.warning(self, "Not saved", msg)

    # receive a sample from serial thread
    def on_sample_ready(self, sample: dict):
        self.recorder.add_sample(
            local_timestamp=sample["local_timestamp"],
            rssi=sample["rssi"],
            noise_floor=sample["noise_floor"],
            channel=sample["channel"],
            fft_gain=sample["fft_gain"],
            agc_gain=sample["agc_gain"],
            amps=sample["amps"],
        )
        # update status occasionally (cheap)
        if len(self.recorder.samples) % 200 == 0 and len(self.recorder.samples) > 0:
            self.status.setText(self._status_text())

    # keyboard shortcuts inside the Qt window (no root)
    def keyPressEvent(self, event):
        k = event.key()
        if k == QtCore.Qt.Key_0:
            self.set_label(0)
        elif k == QtCore.Qt.Key_1:
            self.set_label(1)
        elif k == QtCore.Qt.Key_2:
            self.set_label(2)
        elif k == QtCore.Qt.Key_R:
            self.toggle_recording()
        elif k == QtCore.Qt.Key_S:
            self.save_data()
        else:
            super().keyPressEvent(event)

    # --------- plot update ----------
    def update_curve_colors(self, color_list):
        self.deta_len = len(color_list)
        self.iq_colors = color_list
        self.plotWidget_ted.setXRange(0, max(1, self.deta_len // 2))
        # update subcarrier curve pens (only for existing indices)
        limit = min(self.deta_len, len(self.curve_list), len(self.curve_phase_list))
        for i in range(limit):
            self.curve_list[i].setPen(color_list[i])
            self.curve_phase_list[i].setPen(color_list[i])

    def update_data(self):
        i = np.real(csi_data_complex[-1, :])
        q = np.imag(csi_data_complex[-1, :])

        points = []
        for idx in range(min(self.deta_len, len(i))):
            color = self.iq_colors[idx] if idx < len(self.iq_colors) else (200, 200, 200)
            points.append({'pos': (i[idx], q[idx]), 'brush': pg.mkBrush(color)})
        self.iq_scatter.setData(points)

        self.csi_amplitude_array = np.abs(csi_data_complex)
        self.csi_phase_array = np.angle(csi_data_complex)
        self.csi_row_data = self.csi_phase_array[-1, :]

        self.curve.setData(self.csi_row_data)

        # NOTE: curve_list[0]=AGC, [1]=FFT, subcarriers start at [2]
        self.curve_list[0].setData(agc_gain_data)
        self.curve_list[1].setData(fft_gain_data)

        for sc in range(CSI_DATA_COLUMNS):
            self.curve_list[sc + 2].setData(self.csi_amplitude_array[:, sc])
            self.curve_phase_list[sc].setData(self.csi_phase_array[:, sc])


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print('Python version should >= 3.6')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Read CSI data from serial port and display it graphically + collect labeled dataset')
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help='Serial port (e.g. /dev/ttyACM0)')
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help='Save raw serial CSV to a file (unchanged behavior)')
    parser.add_argument('-l', '--log', dest='log_file', action='store', default='./csi_data_log.txt',
                        help='Save bad CSI data / other serial logs to a log file')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    subthread = SubThread(args.port, args.store_file, args.log_file)

    window = csi_data_graphical_window()
    subthread.data_ready.connect(window.update_curve_colors)
    subthread.sample_ready.connect(window.on_sample_ready)

    subthread.start()
    window.show()

    sys.exit(app.exec())
