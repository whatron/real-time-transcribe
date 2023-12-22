import asyncio
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import soundfile as sf
import time
import threading
import queue

# TODO:
# !!recheck imports & add requirements.txt
# Priority: High
#       Split audio in transcriber rather than recorder
#       ?Silence to split buffer (in addition to) rather than buffer size
#       refactor code
# Seperate Terminal Transcribe and Transcribe library - 2 repos
#       add way to change model size
#       add comments


def record_buffer(samplerate, audio_queue, buffer_duration, stop_event, **kwargs):
    idx = 0
    buffer = np.empty(((samplerate * buffer_duration), 1), dtype="float32")

    def callback(indata, frame_count, time_info, status):
        nonlocal idx
        nonlocal buffer
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            audio_queue.put(buffer)
            buffer = np.empty(
                (samplerate * buffer_duration, 1), dtype="float32")
            idx = 0
        indata = indata[:remainder]
        buffer[idx:idx + len(indata)] = indata
        idx += len(indata)

    with sd.InputStream(callback=callback, dtype=buffer.dtype,
                        channels=buffer.shape[1], samplerate=samplerate, **kwargs):
        try:
            while not stop_event.is_set():
                sd.sleep(1)  # Sleep to reduce CPU usage
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping recording...")
        finally:
            audio_queue.put(buffer[:idx])
            stop_event.set()


def load_model(model_size="base"):
    if len(sys.argv) == 2:
        if sys.argv[1] not in ["tiny", "tiny.en", "small", "small.en", "base", "base.en" "medium", "medium.en", "large-v1", "large-v2", "large-v3", "xl"]:
            raise Exception("Invalid model")
        model_size = {sys.argv[1]}
    elif len(sys.argv) > 2:
        raise Exception("Too many arguments")

    try:
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("Running on GPU with FP16")
    except:
        try:
            # or run on GPU with INT8
            model = WhisperModel(model_size, device="cuda",
                                 compute_type="int8_float16")
            print("Running on GPU with INT8")
        except:
            try:
                # Run on CPU with FP32
                model = WhisperModel(
                    model_size, device="cpu", compute_type="float32")
                print("Running on CPU with FP32")
            except:
                try:
                    # or run on CPU with INT8
                    model = WhisperModel(
                        model_size, device="cpu", compute_type="int8")
                    print("Running on CPU with INT8")
                except:
                    raise Exception("No supported device found")
    return model


async def transcribe(model, audio):

    segments, info = model.transcribe(
        audio, beam_size=5, vad_filter=True, word_timestamps=False)

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    word_list = []

    for segment in segments:
        print(segment.text, end="", flush=True)

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # for segment in segments:
    #     for word in segment.words:
    #         word_list.append(word)
    #         print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

    # print(word_list)


def dequeue_to_list(q):
    return [q.get() for _ in range(q.qsize())]


async def main(samplerate=8000, model=load_model(), channels=1, dtype='float32', buffer_duration=3, **kwargs):
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    print('Recording ...')
    recording_thread = threading.Thread(target=record_buffer, args=(
        samplerate, audio_queue, buffer_duration, stop_event), kwargs=kwargs)
    recording_thread.start()

    time.sleep(5)
    print("Transcript:")

    while True:
        try:
            # sleep for a short amount of time to reduce CPU usage
            time.sleep(0.5)
            # append buffers, clear queue
            queue_length = audio_queue.qsize()
            if queue_length > 0:
                start_time = time.time()
                if queue_length > 1:
                    buffer = np.concatenate(
                        buffer, dequeue_to_list(audio_queue), axis=0)
                else:
                    buffer = audio_queue.get()
                sf.write('buffer.wav', buffer, samplerate)
                # TODO: figure out why this does not work
                # await transcribe(model, buffer.flatten())
                await transcribe(model, 'buffer.wav')
                end_time = time.time()
                # execution_time = end_time - start_time
                # print("Transcribe execution time: %.2f seconds" % execution_time)
        except KeyboardInterrupt:
            print("\nStopping...")
            stop_event.set()
            recording_thread.join()
            raise KeyboardInterrupt
        except Exception as e:
            print("\n\nError occurred")
            print(e)
            raise KeyboardInterrupt

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Sucessfully stopped")
        sys.exit('\nInterrupted by user\n')
    except Exception as e:
        print(e)
        sys.exit('\nError occurred\n')
