import argparse
import subprocess
import threading
import time
import re
import sys

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    print("Tkinter not available:", e)
    sys.exit(1)

try:
    import torch
    CUDA_ENABLED = torch.cuda.is_available()
except Exception:
    CUDA_ENABLED = False


class RunnerGUI:
    def __init__(self, cmd: str):
        self.cmd = cmd
        self.proc = None
        self.root = tk.Tk()
        self.root.title('TextFooler Attack Runner')
        self.root.geometry('800x420')
        self._build_ui()
        # Emit GPU/CPU info at startup
        try:
            self._emit_gpu_info()
        except Exception as e:
            try:
                print('GPU_INFO startup error:', e)
            except Exception:
                pass
            self._append_log('GPU_INFO startup error: ' + str(e))
        self.current_task = '-'
        self.progress_current = 0
        self.progress_total = 0
        self._stop_reader = False

    def _build_ui(self):
        pad = {'padx': 10, 'pady': 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill='both', expand=True)

        frm.columnconfigure(0, weight=1)

        self.lbl_cuda = ttk.Label(frm, text=f'CUDA: {"Enabled" if CUDA_ENABLED else "Disabled"}', font=('Segoe UI', 11))
        self.lbl_cuda.grid(row=0, column=0, sticky='w', **pad)

        # Hardware info (stacked vertically to avoid overflow)
        self.lbl_cpu = ttk.Label(frm, text='CPU: -', font=('Segoe UI', 10))
        self.lbl_cpu.grid(row=1, column=0, sticky='w', **pad)
        self.lbl_gpu = ttk.Label(frm, text='GPU: -', font=('Segoe UI', 10))
        self.lbl_gpu.grid(row=2, column=0, sticky='w', **pad)

        self.lbl_task = ttk.Label(frm, text='Task: -', font=('Segoe UI', 11))
        self.lbl_task.grid(row=3, column=0, sticky='w', **pad)

        self.lbl_prog = ttk.Label(frm, text='Progress: 0 / 0 (0%)', font=('Segoe UI', 10))
        self.lbl_prog.grid(row=4, column=0, sticky='w', **pad)

        self.bar = ttk.Progressbar(frm, orient='horizontal', mode='determinate')
        self.bar.grid(row=5, column=0, sticky='we', **pad)

        # Sub-step
        self.lbl_sub = ttk.Label(frm, text='Step: -', font=('Segoe UI', 10))
        self.lbl_sub.grid(row=6, column=0, sticky='w', **pad)

        self.lbl_subprog = ttk.Label(frm, text='Step Progress: 0 / 0 (0%)', font=('Segoe UI', 9))
        self.lbl_subprog.grid(row=7, column=0, sticky='w', **pad)

        self.bar2 = ttk.Progressbar(frm, orient='horizontal', mode='determinate')
        self.bar2.grid(row=8, column=0, sticky='we', **pad)

        # Tuned params
        self.lbl_params = ttk.Label(frm, text='Params: -', font=('Segoe UI', 9))
        self.lbl_params.grid(row=9, column=0, sticky='w', **pad)

        # Log with scrollbar
        self.txt = tk.Text(frm, height=10, wrap='word')
        self.txt.grid(row=10, column=0, sticky='nsew', **pad)
        scroll = ttk.Scrollbar(frm, orient='vertical', command=self.txt.yview)
        scroll.grid(row=10, column=1, sticky='ns', **pad)
        self.txt.configure(yscrollcommand=scroll.set)
        frm.rowconfigure(10, weight=1)

    def _append_log(self, line: str):
        self.txt.insert('end', line + '\n')
        self.txt.see('end')

    def _update_task(self, task: str):
        self.current_task = task
        self.lbl_task.config(text=f'Task: {task}')

    def _update_progress(self, cur: int, tot: int):
        self.progress_current = cur
        self.progress_total = tot
        pct = 0.0
        if tot:
            pct = round(100.0 * float(cur) / float(tot), 2)
        self.lbl_prog.config(text=f'Progress: {cur} / {tot} ({pct}%)')
        self.bar['maximum'] = max(1, tot)
        self.bar['value'] = min(tot, cur)

    def _update_substep(self, name: str, cur: int, tot: int):
        self.lbl_sub.config(text=f'Step: {name}')
        pct = 0.0
        if tot:
            pct = round(100.0 * float(cur) / float(tot), 2)
        self.lbl_subprog.config(text=f'Step Progress: {cur} / {tot} ({pct}%)')
        self.bar2['maximum'] = max(1, tot)
        self.bar2['value'] = min(tot, cur)

    def _parse_line(self, line: str):
        s = line.strip()
        if not s:
            return
        # Update task by known markers
        if 'Data import finished!' in s:
            self._update_task('Import data')
        elif 'Building Model...' in s:
            self._update_task('Build model')
        elif 'Model built!' in s:
            self._update_task('Model built')
        elif 'Building vocab...' in s:
            self._update_task('Build vocab')
        elif 'Building cos sim matrix...' in s:
            self._update_task('Build cosine similarity')
        elif 'Cos sim import finished!' in s:
            self._update_task('Cos sim ready')
        elif 'Start attacking!' in s:
            self._update_task('Attacking')
        # Progress line: "123 samples out of 1000 have been finished!"
        m = re.search(r"(\d+)\s+samples\s+out\s+of\s+(\d+)", s)
        if m:
            cur = int(m.group(1))
            tot = int(m.group(2))
            self._update_progress(cur, tot)
        # Hardware and tuning markers
        if s.startswith('HW_CPU '):
            m = re.match(r'^HW_CPU\s+name=(.+)\s+logical=(\d+)', s)
            if m:
                name = m.group(1)
                logical = m.group(2)
                self.lbl_cpu.config(text=f'CPU: {name} ({logical} threads)')
        elif s.startswith('HW_GPU '):
            m = re.match(r'^HW_GPU\s+name=(.+)\s+mem_gb=([0-9.]+)\s+cc=(.+)', s)
            if m:
                name = m.group(1)
                mem = m.group(2)
                cc = m.group(3)
                self.lbl_gpu.config(text=f'GPU: {name} ({mem} GB, cc {cc})')
        elif s.startswith('TUNING '):
            self.lbl_params.config(text=f'Params: {s[7:]}')
        # Hardware and tuning markers
        if s.startswith('HW_CPU '):
            m = re.match(r'^HW_CPU\s+name=(.+)\s+logical=(\d+)', s)
            if m:
                name = m.group(1)
                logical = m.group(2)
                self.lbl_cpu.config(text=f'CPU: {name} ({logical} threads)')
        elif s.startswith('HW_GPU '):
            m = re.match(r'^HW_GPU\s+name=(.+)\s+mem_gb=([0-9.]+)\s+cc=(.+)', s)
            if m:
                name = m.group(1)
                mem = m.group(2)
                cc = m.group(3)
                self.lbl_gpu.config(text=f'GPU: {name} ({mem} GB, cc {cc})')
        elif s.startswith('TUNING '):
            self.lbl_params.config(text=f'Params: {s[7:]}')
        # Sub-step markers
        if s.startswith('STEP_START '):
            # e.g., STEP_START Importance scoring total=100
            m = re.match(r'^STEP_START\s+(.+?)\s+total=(\d+)', s)
            if m:
                name = m.group(1)
                tot = int(m.group(2))
                self._update_substep(name, 0, tot)
        elif s.startswith('STEP_PROGRESS '):
            # e.g., STEP_PROGRESS Importance scoring 12/100
            m = re.match(r'^STEP_PROGRESS\s+(.+?)\s+(\d+)/(\d+)', s)
            if m:
                name = m.group(1)
                cur = int(m.group(2))
                tot = int(m.group(3))
                self._update_substep(name, cur, tot)
        elif s.startswith('STEP_DONE '):
            m = re.match(r'^STEP_DONE\s+(.+)', s)
            if m:
                name = m.group(1)
                # mark done
                self._update_substep(name, self.bar2['maximum'], self.bar2['maximum'])

    def _emit_gpu_info(self):
        """Print GPU availability and counts, referencing TensorFlow_test outputs."""
        lines = []
        # Torch info
        try:
            import torch as _torch
            torch_avail = _torch.cuda.is_available()
            torch_count = _torch.cuda.device_count() if torch_avail else 0
            lines.append(f"GPU_INFO Torch CUDA available: {torch_avail}")
            lines.append(f"GPU_INFO Torch Num GPUs Available: {torch_count}")
            for i in range(torch_count):
                try:
                    name = _torch.cuda.get_device_name(i)
                    lines.append(f"GPU_INFO Torch GPU[{i}] name: {name}")
                except Exception:
                    pass
        except Exception as e:
            lines.append(f"GPU_INFO Torch not available: {e}")

        # TensorFlow info
        try:
            import tensorflow as tf
            tf_version = getattr(tf, '__version__', 'unknown')
            lines.append(f"GPU_INFO TF Version: {tf_version}")
            try:
                dev_name = tf.test.gpu_device_name()
                lines.append(f"GPU_INFO TF gpu_device_name(): {dev_name}")
            except Exception as e1:
                lines.append(f"GPU_INFO TF gpu_device_name() error: {e1}")
            try:
                phys_gpus = tf.config.list_physical_devices('GPU')
                lines.append(f"GPU_INFO TF Physical GPUs: {phys_gpus}")
                lines.append(f"GPU_INFO TF Num GPUs Available: {len(phys_gpus)}")
            except Exception as e2:
                lines.append(f"GPU_INFO TF Physical GPUs error: {e2}")
            try:
                is_avail = tf.test.is_gpu_available()
                lines.append(f"GPU_INFO TF is_gpu_available(): {is_avail}")
            except Exception as e3:
                lines.append(f"GPU_INFO TF is_gpu_available() error: {e3}")
        except Exception as e:
            lines.append(f"GPU_INFO TF not available: {e}")

        # Emit to console and GUI log
        for ln in lines:
            try:
                print(ln)
            except Exception:
                pass
            self._append_log(ln)

        # Update CUDA label with GPU count (prefer torch, fallback to tf)
        gpu_count = 0
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                gpu_count = _torch.cuda.device_count()
        except Exception:
            pass
        if gpu_count == 0:
            try:
                import tensorflow as tf
                gpu_count = len(tf.config.list_physical_devices('GPU'))
            except Exception:
                gpu_count = 0
        self.lbl_cuda.config(text=f'CUDA: {"Enabled" if CUDA_ENABLED else "Disabled"} (GPUs: {gpu_count})')

    def _reader(self):
        with self.proc.stdout:
            for raw in iter(self.proc.stdout.readline, ''):
                if self._stop_reader:
                    break
                line = raw.rstrip('\n')
                # Echo to terminal
                try:
                    print(line)
                except Exception:
                    pass
                # Update GUI
                self.root.after(0, self._append_log, line)
                self.root.after(0, self._parse_line, line)
        # Consume remaining stderr
        try:
            err = self.proc.stderr.read()
            if err:
                for ln in err.splitlines():
                    try:
                        print('[ERR] ' + ln)
                    except Exception:
                        pass
                    self.root.after(0, self._append_log, '[ERR] ' + ln)
        except Exception:
            pass
        # Done
        self.root.after(0, self._update_task, 'Done')

    def run(self):
        # Launch subprocess
        # Use unbuffered python (-u) via caller, but also request line-buffered pipe here
        self.proc = subprocess.Popen(self.cmd,
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     bufsize=1,
                                     text=True,
                                     encoding='utf-8',
                                     errors='replace')
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.mainloop()
        self._stop_reader = True
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    def _on_close(self):
        self._stop_reader = True
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmd', type=str, required=True, help='Command to run attack script')
    args = ap.parse_args()
    gui = RunnerGUI(args.cmd)
    gui.run()


if __name__ == '__main__':
    main()