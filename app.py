import asyncio
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import soundfile as sf
import time
import threading
import queue

def record_buffer(buffer, samplerate, audio_queue, **kwargs):
    idx = 0

    def callback(indata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            raise sd.CallbackStop
        indata = indata[:remainder]
        buffer[idx:idx + len(indata)] = indata
        idx += len(indata)

    with sd.InputStream(callback=callback, dtype=buffer.dtype,
                        channels=buffer.shape[1], samplerate=samplerate, **kwargs):
        try:
            while True:
                time.sleep(0.1)  # sleep for a short amount of time to reduce CPU usage
                if idx >= len(buffer):
                    audio_queue.put(buffer)
                    print('buffer reset')
                    buffer = np.empty((samplerate * 20, 1), dtype="float32")
                    idx = 0
        except KeyboardInterrupt:
            # close the stream on KeyboardInterrupt
            pass

def load_model(model_size="base"):
    try:
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("Running on GPU with FP16")
    except:
        try:
            # Run on CPU with FP32
            model = WhisperModel(model_size, device="cpu", compute_type="float32")
            print("Running on CPU with FP32")
        except:
            try:
                # or run on GPU with INT8
                model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                print("Running on GPU with INT8")
            except:
                try:
                    # or run on CPU with INT8
                    model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print("Running on CPU with INT8")
                except:
                    raise Exception("No supported device found")
    return model


async def transcribe(model, audio):

    segments, info = model.transcribe(audio, beam_size=5, vad_filter=True, word_timestamps=True)

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    word_list = []

    for segment in segments:
        for word in segment.words:
            word_list.append(word)
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            
    # print(word_list)
    
def dequeue_to_list(q):
    result_list = []
    while not q.empty():
        item = q.get()
        result_list.append(item)
    return result_list

async def main(samplerate=8000, model=load_model(), channels=1, dtype='float32', **kwargs):
    audio_queue = queue.Queue()
    buffer = np.empty((samplerate * 5, channels), dtype=dtype)
    print('recording ...')
    recording_thread = threading.Thread(target=record_buffer, args=(buffer, samplerate, audio_queue), kwargs=kwargs)
    recording_thread.start()

    time.sleep(5)

    while True:
        try:
            # append buffers, clear queue
            queue_length = audio_queue.qsize()
            if queue_length > 0:
                start_time = time.time()
                if queue_length > 1:
                    buffer = np.concatenate(buffer, dequeue_to_list(audio_queue), axis=0)
                else:
                    buffer = audio_queue.get()
                sf.write('op.wav', buffer, samplerate)
                await transcribe(model, 'op.wav')
                end_time = time.time()
                execution_time = end_time - start_time
                print("Transcribe execution time: %.2f seconds" % execution_time)
        except KeyboardInterrupt:
            print('interrupted!')
            break
        except:
            print("Error occured")
            break

    # TODO: figure out why this does not work
    # await transcribe(model, buffer.flatten())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
