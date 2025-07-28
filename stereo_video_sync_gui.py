import sys
import os
import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import subprocess
import uuid
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSlider, QLineEdit, QTextEdit, QGroupBox, 
                            QMessageBox, QProgressBar, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

class MplCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class AudioProcessingThread(QThread):
    """Thread for processing audio to avoid freezing the UI"""
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, threshold, max_peaks, start_time=0, end_time=None):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.max_peaks = max_peaks
        self.start_time = start_time  # Start time in seconds
        self.end_time = end_time      # End time in seconds (None = end of video)
    
    def run(self):
        try:
            # Create temp audio file with unique identifier
            self.progress_update.emit(10)
            unique_id = uuid.uuid4().hex[:8]
            temp_wav = f"temp_audio_{unique_id}_{os.path.basename(self.video_path)}.wav"
            
            # Get video info first
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            # Handle full video case
            if self.start_time is None:
                self.start_time = 0
            if self.end_time is None:
                self.end_time = duration
            
            # Validate time window
            if self.end_time > duration:
                self.end_time = duration
            if self.start_time >= self.end_time:
                self.error_occurred.emit("Start time must be less than end time")
                return
            
            # Extract audio with time window
            self.progress_update.emit(20)
            if not self.extract_audio_segment(self.video_path, temp_wav, self.start_time, self.end_time):
                self.error_occurred.emit(f"Failed to extract audio from {self.video_path}")
                return
            
            # Read the audio file
            self.progress_update.emit(40)
            try:
                sample_rate, audio_data = wavfile.read(temp_wav)
            except Exception as e:
                self.error_occurred.emit(f"Error reading audio file: {e}")
                try:
                    os.remove(temp_wav)
                except:
                    pass
                return
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Get absolute values
            abs_audio = np.abs(audio_data)
            
            # Calculate threshold
            mean_value = np.mean(abs_audio)
            threshold = mean_value * self.threshold
            
            self.progress_update.emit(60)
            # Find peaks
            min_separation = int(sample_rate * 0.5)  # 0.5 seconds
            potential_peaks, properties = find_peaks(abs_audio, height=threshold, distance=min_separation)
            
            # Sort peaks by amplitude and take the top peaks
            peak_amplitudes = abs_audio[potential_peaks]
            sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Sort in descending order
            
            # Take only the top peaks
            if len(sorted_indices) > self.max_peaks:
                sorted_indices = sorted_indices[:self.max_peaks]
            
            # Get the final peak indices, sorted by their position in the audio
            peaks_indices = sorted(potential_peaks[sorted_indices])
            
            if not peaks_indices:
                self.error_occurred.emit(f"No peaks found above threshold {self.threshold} in the specified time window. Try lowering the threshold.")
                try:
                    os.remove(temp_wav)
                except:
                    pass
                return
            
            self.progress_update.emit(80)
            
            # Convert peaks to times and frames (relative to start of video, not segment)
            segment_duration = self.end_time - self.start_time
            peak_times_in_segment = [idx / sample_rate for idx in peaks_indices]
            peak_times_absolute = [t + self.start_time for t in peak_times_in_segment]  # Absolute time in video
            peak_values = [abs_audio[idx] for idx in peaks_indices]
            peak_frames_absolute = [int(time * fps) for time in peak_times_absolute]  # Absolute frame numbers
            
            # Prepare plot data
            plot_data = {
                'abs_audio': abs_audio,
                'sample_rate': sample_rate,
                'peak_times': peak_times_in_segment,  # For plotting (relative to segment)
                'peak_times_absolute': peak_times_absolute,  # For calculations (absolute)
                'peak_values': peak_values,
                'peak_frames': peak_frames_absolute,  # Absolute frame numbers for sync
                'fps': fps,
                'threshold': threshold,
                'duration': duration,
                'segment_duration': segment_duration,
                'start_time': self.start_time,
                'end_time': self.end_time
            }
            
            self.progress_update.emit(90)
            # Clean up
            try:
                os.remove(temp_wav)
            except:
                pass
            
            self.progress_update.emit(100)
            self.processing_complete.emit(plot_data)
            
        except Exception as e:
            self.error_occurred.emit(f"An error occurred: {str(e)}")
    
    def extract_audio_segment(self, video_path, output_audio_path, start_time, end_time):
        """Extract audio segment from video file using ffmpeg"""
        duration = end_time - start_time
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),     # Start time
            '-t', str(duration),        # Duration
            '-q:a', '0',
            '-map', 'a',
            '-y',  # Overwrite output file if it exists
            output_audio_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            self.error_occurred.emit(f"Error extracting audio: {e}")
            return False

class AutoSyncThread(QThread):
    """Thread for automatically calculating sync between videos"""
    sync_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, left_data, right_data):
        super().__init__()
        self.left_data = left_data
        self.right_data = right_data
    
    def run(self):
        try:
            left_peaks = self.left_data['peak_frames']
            right_peaks = self.right_data['peak_frames']
            left_fps = self.left_data['fps']
            right_fps = self.right_data['fps']
            
            # Calculate the frame offset automatically
            frame_offset = self.auto_match_peaks(left_peaks, right_peaks, left_fps, right_fps)
            
            if frame_offset is None:
                self.error_occurred.emit("Automatic sync failed. Try adjusting the threshold or manually selecting peaks.")
                return
            
            result = {
                'frame_offset': frame_offset,
                'left_fps': left_fps,
                'right_fps': right_fps
            }
            
            self.sync_complete.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(f"An error occurred during sync: {str(e)}")
    
    def auto_match_peaks(self, left_peaks, right_peaks, left_fps, right_fps, tolerance_frames=2):
        """Find matching peaks between left and right videos"""
        
        # Calculate time difference patterns between consecutive peaks in each video
        left_diffs = [left_peaks[i+1] - left_peaks[i] for i in range(len(left_peaks)-1)]
        right_diffs = [right_peaks[i+1] - right_peaks[i] for i in range(len(right_peaks)-1)]
        
        # Convert to frame differences
        left_frame_diffs = left_diffs
        right_frame_diffs = right_diffs
        
        # Find matching patterns between videos
        matches = []
        
        # Try to find at least 3 consecutive peaks with similar intervals
        for i in range(len(left_frame_diffs) - 2):
            for j in range(len(right_frame_diffs) - 2):
                # Check if three consecutive intervals match within tolerance
                match = True
                for k in range(3):
                    if i+k < len(left_frame_diffs) and j+k < len(right_frame_diffs):
                        # Check if the frame difference patterns match within tolerance
                        left_diff = left_frame_diffs[i+k]
                        right_diff = right_frame_diffs[j+k]
                        ratio = left_diff / right_diff if right_diff != 0 else float('inf')
                        
                        # If the ratio is not close to 1, it's not a match
                        if ratio < 0.9 or ratio > 1.1:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    matches.append((i, j))
        
        if not matches:
            # If no pattern matches, try direct frame offsets
            possible_offsets = []
            for i, left_frame in enumerate(left_peaks):
                for j, right_frame in enumerate(right_peaks):
                    offset = right_frame - left_frame
                    possible_offsets.append(offset)
            
            # Find the most common offset (with some tolerance)
            if possible_offsets:
                # Group similar offsets
                grouped_offsets = {}
                for offset in possible_offsets:
                    matched = False
                    for key in grouped_offsets:
                        if abs(offset - key) <= tolerance_frames:
                            grouped_offsets[key] += 1
                            matched = True
                            break
                    if not matched:
                        grouped_offsets[offset] = 1
                
                # Find the most common offset
                most_common_offset = max(grouped_offsets.items(), key=lambda x: x[1])
                offset = most_common_offset[0]
                count = most_common_offset[1]
                
                # Only use if we have at least 2 matches
                if count >= 2:
                    return offset
                else:
                    return None
            else:
                return None
        
        # If we have pattern matches
        best_match = matches[0]  # Take the first match
        i, j = best_match
        
        # Calculate the offset using the matching peaks
        left_start_idx = i
        right_start_idx = j
        
        offset = right_peaks[right_start_idx] - left_peaks[left_start_idx]
        
        return offset

class VR180SyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VR180 Sync Tool")
        self.setGeometry(100, 100, 1200, 900)
        
        self.left_video_path = ""
        self.right_video_path = ""
        self.left_data = None
        self.right_data = None
        
        self.initUI()
    
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # File selection area
        file_group = QGroupBox("Video File Selection")
        file_layout = QVBoxLayout()
        
        # Left video selection
        left_layout = QHBoxLayout()
        left_label = QLabel("Left Video:")
        self.left_path_edit = QLineEdit()
        self.left_path_edit.setReadOnly(True)
        left_browse_btn = QPushButton("Browse...")
        left_browse_btn.clicked.connect(lambda: self.browse_file('left'))
        
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_path_edit)
        left_layout.addWidget(left_browse_btn)
        
        # Right video selection
        right_layout = QHBoxLayout()
        right_label = QLabel("Right Video:")
        self.right_path_edit = QLineEdit()
        self.right_path_edit.setReadOnly(True)
        right_browse_btn = QPushButton("Browse...")
        right_browse_btn.clicked.connect(lambda: self.browse_file('right'))
        
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.right_path_edit)
        right_layout.addWidget(right_browse_btn)
        
        file_layout.addLayout(left_layout)
        file_layout.addLayout(right_layout)
        file_group.setLayout(file_layout)
        
        # Time window selection area
        time_window_group = QGroupBox("Analysis Time Window")
        time_window_layout = QVBoxLayout()
        
        # Instructions
        instruction_label = QLabel("Set the time window to analyze (useful for clap cues at start/end of video):")
        instruction_label.setStyleSheet("font-style: italic; color: #666;")
        time_window_layout.addWidget(instruction_label)
        
        # Time inputs
        time_inputs_layout = QHBoxLayout()
        
        # Start time
        start_time_layout = QVBoxLayout()
        start_time_label = QLabel("Start Time (seconds):")
        self.start_time_spinbox = QSpinBox()
        self.start_time_spinbox.setMinimum(0)
        self.start_time_spinbox.setMaximum(3600)  # Max 1 hour
        self.start_time_spinbox.setValue(0)
        self.start_time_spinbox.setSuffix(" sec")
        
        start_time_layout.addWidget(start_time_label)
        start_time_layout.addWidget(self.start_time_spinbox)
        
        # End time
        end_time_layout = QVBoxLayout()
        end_time_label = QLabel("End Time (seconds):")
        self.end_time_spinbox = QSpinBox()
        self.end_time_spinbox.setMinimum(1)
        self.end_time_spinbox.setMaximum(3600)  # Max 1 hour
        self.end_time_spinbox.setValue(30)  # Default to first 30 seconds
        self.end_time_spinbox.setSuffix(" sec")
        
        end_time_layout.addWidget(end_time_label)
        end_time_layout.addWidget(self.end_time_spinbox)
        
        # Use full video checkbox
        self.use_full_video_checkbox = QCheckBox("Use full video duration")
        self.use_full_video_checkbox.toggled.connect(self.toggle_time_window)
        
        time_inputs_layout.addLayout(start_time_layout)
        time_inputs_layout.addLayout(end_time_layout)
        time_inputs_layout.addWidget(self.use_full_video_checkbox)
        
        time_window_layout.addLayout(time_inputs_layout)
        time_window_group.setLayout(time_window_layout)
        
        # Settings area
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QHBoxLayout()
        
        # Threshold slider
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(60)
        self.threshold_slider.setValue(30)  # Default 3.0
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_value_label = QLabel("3.0")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        
        # Max peaks input
        max_peaks_layout = QVBoxLayout()
        max_peaks_label = QLabel("Max Peaks:")
        self.max_peaks_edit = QLineEdit("15")
        
        max_peaks_layout.addWidget(max_peaks_label)
        max_peaks_layout.addWidget(self.max_peaks_edit)
        
        settings_layout.addLayout(threshold_layout)
        settings_layout.addLayout(max_peaks_layout)
        settings_group.setLayout(settings_layout)
        
        # Process buttons
        process_layout = QHBoxLayout()
        self.process_left_btn = QPushButton("Process Left Video")
        self.process_left_btn.clicked.connect(self.process_left_video)
        self.process_left_btn.setEnabled(False)
        
        self.process_right_btn = QPushButton("Process Right Video")
        self.process_right_btn.clicked.connect(self.process_right_video)
        self.process_right_btn.setEnabled(False)
        
        self.auto_sync_btn = QPushButton("Calculate Sync")
        self.auto_sync_btn.clicked.connect(self.calculate_sync)
        self.auto_sync_btn.setEnabled(False)
        
        process_layout.addWidget(self.process_left_btn)
        process_layout.addWidget(self.process_right_btn)
        process_layout.addWidget(self.auto_sync_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Plot area
        plot_layout = QHBoxLayout()
        
        # Left plot
        left_plot_group = QGroupBox("Left Video Audio Peaks")
        left_plot_layout = QVBoxLayout()
        self.left_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        left_plot_layout.addWidget(self.left_canvas)
        left_plot_group.setLayout(left_plot_layout)
        
        # Right plot
        right_plot_group = QGroupBox("Right Video Audio Peaks")
        right_plot_layout = QVBoxLayout()
        self.right_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_plot_layout.addWidget(self.right_canvas)
        right_plot_group.setLayout(right_plot_layout)
        
        plot_layout.addWidget(left_plot_group)
        plot_layout.addWidget(right_plot_group)
        
        # Results area
        results_group = QGroupBox("Sync Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        # Add everything to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(time_window_group)
        main_layout.addWidget(settings_group)
        main_layout.addLayout(process_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(plot_layout)
        main_layout.addWidget(results_group)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def toggle_time_window(self, checked):
        """Enable/disable time window controls"""
        self.start_time_spinbox.setEnabled(not checked)
        self.end_time_spinbox.setEnabled(not checked)
    
    def update_threshold_label(self):
        """Update the threshold value label when slider changes"""
        value = self.threshold_slider.value() / 10.0
        self.threshold_value_label.setText(f"{value:.1f}")
    
    def browse_file(self, side):
        """Open file dialog to select video file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {side.capitalize()} Video", "", 
            "Video Files (*.mp4 *.mov *.avi);;All Files (*)", 
            options=options
        )
        
        if file_path:
            if side == 'left':
                self.left_video_path = file_path
                self.left_path_edit.setText(file_path)
                self.process_left_btn.setEnabled(True)
                
                # Update time window based on video duration
                self.update_time_window_limits(file_path)
            else:
                self.right_video_path = file_path
                self.right_path_edit.setText(file_path)
                self.process_right_btn.setEnabled(True)
            
            # Enable auto sync button if both videos are selected and processed
            if self.left_data and self.right_data:
                self.auto_sync_btn.setEnabled(True)
    
    def update_time_window_limits(self, video_path):
        """Update time window spinbox limits based on video duration"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(frame_count / fps)
            cap.release()
            
            self.end_time_spinbox.setMaximum(duration)
            if self.end_time_spinbox.value() > duration:
                self.end_time_spinbox.setValue(min(30, duration))
            
        except Exception as e:
            print(f"Error getting video duration: {e}")
    
    def get_time_window(self):
        """Get the current time window settings"""
        if self.use_full_video_checkbox.isChecked():
            return None, None  # Use full video
        else:
            start_time = self.start_time_spinbox.value()
            end_time = self.end_time_spinbox.value()
            return start_time, end_time
    
    def process_left_video(self):
        """Process the left video to find audio peaks"""
        if not self.left_video_path:
            QMessageBox.warning(self, "Warning", "Please select a left video file first.")
            return
        
        threshold = self.threshold_slider.value() / 10.0
        try:
            max_peaks = int(self.max_peaks_edit.text())
        except:
            max_peaks = 15
        
        start_time, end_time = self.get_time_window()
        
        # Start processing thread
        self.left_thread = AudioProcessingThread(self.left_video_path, threshold, max_peaks, start_time, end_time)
        self.left_thread.progress_update.connect(self.update_progress)
        self.left_thread.processing_complete.connect(self.left_processing_complete)
        self.left_thread.error_occurred.connect(self.show_error)
        
        self.process_left_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.left_thread.start()
    
    def process_right_video(self):
        """Process the right video to find audio peaks"""
        if not self.right_video_path:
            QMessageBox.warning(self, "Warning", "Please select a right video file first.")
            return
        
        threshold = self.threshold_slider.value() / 10.0
        try:
            max_peaks = int(self.max_peaks_edit.text())
        except:
            max_peaks = 15
        
        start_time, end_time = self.get_time_window()
        
        # Start processing thread
        self.right_thread = AudioProcessingThread(self.right_video_path, threshold, max_peaks, start_time, end_time)
        self.right_thread.progress_update.connect(self.update_progress)
        self.right_thread.processing_complete.connect(self.right_processing_complete)
        self.right_thread.error_occurred.connect(self.show_error)
        
        self.process_right_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.right_thread.start()
    
    def left_processing_complete(self, data):
        """Handle completion of left video processing"""
        self.left_data = data
        self.update_left_plot(data)
        self.process_left_btn.setEnabled(True)
        
        # Enable auto sync if both videos are processed
        if self.right_data:
            self.auto_sync_btn.setEnabled(True)
    
    def right_processing_complete(self, data):
        """Handle completion of right video processing"""
        self.right_data = data
        self.update_right_plot(data)
        self.process_right_btn.setEnabled(True)
        
        # Enable auto sync if both videos are processed
        if self.left_data:
            self.auto_sync_btn.setEnabled(True)
    
    def update_left_plot(self, data):
        """Update the left plot with processed data"""
        self.left_canvas.axes.clear()
        
        # Plot waveform
        downsample_factor = max(1, len(data['abs_audio']) // 10000)
        time_axis = np.arange(0, len(data['abs_audio']), downsample_factor) / data['sample_rate']
        self.left_canvas.axes.plot(time_axis, data['abs_audio'][::downsample_factor], 'b-', alpha=0.5)
        
        # Plot peaks
        for i, (time, value) in enumerate(zip(data['peak_times'], data['peak_values'])):
            self.left_canvas.axes.plot(time, value, 'ro', markersize=8)
            self.left_canvas.axes.text(time, value*1.1, f"{i+1}: {data['peak_frames'][i]}", 
                     fontsize=8, horizontalalignment='center')
        
        # Plot threshold
        self.left_canvas.axes.axhline(y=data['threshold'], color='r', linestyle='--')
        
        self.left_canvas.axes.set_xlabel('Time (seconds)')
        self.left_canvas.axes.set_ylabel('Audio Amplitude')
        
        # Update title to show time window
        if data['start_time'] > 0 or data['end_time'] < data['duration']:
            title = f"Left Video Audio Peaks ({data['start_time']:.1f}s - {data['end_time']:.1f}s)"
        else:
            title = "Left Video Audio Peaks"
        self.left_canvas.axes.set_title(title)
        
        self.left_canvas.axes.grid(True)
        self.left_canvas.axes.set_xlim(0, data['segment_duration'])
        
        self.left_canvas.fig.tight_layout()
        self.left_canvas.draw()
        
        # Add peak info to results
        time_window_info = ""
        if data['start_time'] > 0 or data['end_time'] < data['duration']:
            time_window_info = f"Analysis Window: {data['start_time']:.1f}s - {data['end_time']:.1f}s\n"
        
        peak_info = f"{time_window_info}Left Video Peak Frames:\n"
        for i, frame in enumerate(data['peak_frames']):
            peak_info += f"Peak {i+1}: Frame {frame}\n"
        
        self.results_text.setText(peak_info)
    
    def update_right_plot(self, data):
        """Update the right plot with processed data"""
        self.right_canvas.axes.clear()
        
        # Plot waveform
        downsample_factor = max(1, len(data['abs_audio']) // 10000)
        time_axis = np.arange(0, len(data['abs_audio']), downsample_factor) / data['sample_rate']
        self.right_canvas.axes.plot(time_axis, data['abs_audio'][::downsample_factor], 'b-', alpha=0.5)
        
        # Plot peaks
        for i, (time, value) in enumerate(zip(data['peak_times'], data['peak_values'])):
            self.right_canvas.axes.plot(time, value, 'ro', markersize=8)
            self.right_canvas.axes.text(time, value*1.1, f"{i+1}: {data['peak_frames'][i]}", 
                     fontsize=8, horizontalalignment='center')
        
        # Plot threshold
        self.right_canvas.axes.axhline(y=data['threshold'], color='r', linestyle='--')
        
        self.right_canvas.axes.set_xlabel('Time (seconds)')
        self.right_canvas.axes.set_ylabel('Audio Amplitude')
        
        # Update title to show time window
        if data['start_time'] > 0 or data['end_time'] < data['duration']:
            title = f"Right Video Audio Peaks ({data['start_time']:.1f}s - {data['end_time']:.1f}s)"
        else:
            title = "Right Video Audio Peaks"
        self.right_canvas.axes.set_title(title)
        
        self.right_canvas.axes.grid(True)
        self.right_canvas.axes.set_xlim(0, data['segment_duration'])
        
        self.right_canvas.fig.tight_layout()
        self.right_canvas.draw()
        
        # Add peak info to results
        current_text = self.results_text.toPlainText()
        
        time_window_info = ""
        if data['start_time'] > 0 or data['end_time'] < data['duration']:
            time_window_info = f"\nAnalysis Window: {data['start_time']:.1f}s - {data['end_time']:.1f}s\n"
        
        peak_info = current_text + f"{time_window_info}Right Video Peak Frames:\n"
        for i, frame in enumerate(data['peak_frames']):
            peak_info += f"Peak {i+1}: Frame {frame}\n"
        
        self.results_text.setText(peak_info)
    
    def calculate_sync(self):
        """Calculate sync between left and right videos"""
        if not self.left_data or not self.right_data:
            QMessageBox.warning(self, "Warning", "Both videos must be processed first.")
            return
        
        # Start sync thread
        self.sync_thread = AutoSyncThread(self.left_data, self.right_data)
        self.sync_thread.sync_complete.connect(self.sync_complete)
        self.sync_thread.error_occurred.connect(self.show_error)
        
        self.auto_sync_btn.setEnabled(False)
        self.sync_thread.start()
    
    def sync_complete(self, result):
        """Handle completion of sync calculation"""
        frame_offset = result['frame_offset']
        
        # Format the results
        results = self.results_text.toPlainText()
        results += "\n\n=== VR180 Sync Results ===\n"
        results += f"Frame offset: {frame_offset} frames\n\n"
        
        results += "Instructions for 180Augen/180Kino:\n"
        if frame_offset > 0:
            results += f"The right video is {frame_offset} frames ahead of the left video.\n"
            results += f"Start the right video from frame 0 and the left video from frame {abs(frame_offset)}.\n"
        elif frame_offset < 0:
            results += f"The left video is {abs(frame_offset)} frames ahead of the right video.\n"
            results += f"Start the left video from frame {abs(frame_offset)} and the right video from frame 0.\n"
        else:
            results += "Both videos are perfectly synchronized! No offset needed.\n"
        
        self.results_text.setText(results)
        self.auto_sync_btn.setEnabled(True)
        
        # Show message box with results
        QMessageBox.information(self, "Sync Complete", 
                              f"The calculated frame offset is {frame_offset} frames.")
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.process_left_btn.setEnabled(True if self.left_video_path else False)
        self.process_right_btn.setEnabled(True if self.right_video_path else False)
        self.auto_sync_btn.setEnabled(True if self.left_data and self.right_data else False)

def main():
    app = QApplication(sys.argv)
    window = VR180SyncApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()