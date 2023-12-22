# Real Time Transcribe

This project utilizes the Whisper ASR (Automatic Speech Recognition) model to transcribe audio input in real-time. The script records audio input, processes it using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and prints the transcribed text.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/whatron/real-time-transcribe
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```
      - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```


3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:

   ```bash
   python transcript.py
   ```

2. The script will start recording audio and transcribing it in real-time.

3. Press `Ctrl+C` twice to stop the script.

## Configuration

### Model Size

By default, the script uses the "base" model size. You can change the model size by passing a command-line argument:

'''bash
python transcript.py large-v2
'''

Supported model sizes (from fastest to slowest): "tiny", "tiny.en", "small", "small.en", "base", "base.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"<br>
If transcription is taking too long, it is reccomended to use a faster model.

## Requirements

Ensure you have the necessary requirements installed by running:

'''bash
pip install -r requirements.txt
'''

For more information about the Whisper ASR model, visit [faster-whisper](https://github.com/SYSTRAN/faster-whisper).