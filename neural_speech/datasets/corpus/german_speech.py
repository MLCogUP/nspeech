# import os
# import xml.etree.ElementTree as etree
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from glob import glob
#
# import numpy as np
# from unidecode import unidecode
#
# from neural_speech.utils import audio
#
#
# def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
#     '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.
#
#       Args:
#         in_dir: The directory where you have downloaded the LJ Speech dataset
#         out_dir: The directory to write the output into
#         num_workers: Optional number of worker processes to parallelize across
#         tqdm: You can optionally pass tqdm to get a nice progress bar
#
#       Returns:
#         A list of tuples describing the training examples. This should be written to train.txt
#     '''
#     # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
#     # can omit it and just call _process_utterance on each input if you want.
#     executor = ProcessPoolExecutor(max_workers=num_workers)
#     futures = []
#     index = 1
#     xml_path_pattern = os.path.join(in_dir, "train", "*.xml")
#     # print("build from path", xml_path_pattern)
#     for xml_path in glob(xml_path_pattern):
#         file_id = os.path.splitext(os.path.basename(xml_path))[0]
#         # print(file_id)
#         wav_path = os.path.join(os.path.dirname(xml_path), "{}_Kinect-RAW.wav".format(file_id))
#         if os.path.exists(wav_path):
#             tree = etree.parse(xml_path)
#             root = tree.getroot()
#             text = unidecode(root.findall("sentence")[0].text)
#
#             futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
#             index += 1
#     return list(filter(None.__ne__, (future.result() for future in tqdm(futures))))
#
#
# def _process_utterance(out_dir, index, wav_path, text):
#     '''Preprocesses a single utterance audio/text pair.
#
#     This writes the mel and linear scale spectrograms to disk and returns a tuple to write
#     to the train.txt file.
#
#     Args:
#       out_dir: The directory to write the spectrograms into
#       index: The numeric index to use in the spectrogram filenames.
#       wav_path: Path to the audio file containing the speech input
#       text: The text spoken in the input audio file
#
#     Returns:
#       A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
#     '''
#     spectrogram_filename = 'germanspeech-spec-%05d.npy' % index
#     spectrogram_path = os.path.join(out_dir, spectrogram_filename)
#     mel_filename = 'germanspeech-mel-%05d.npy' % index
#     mel_path = os.path.join(out_dir, mel_filename)
#     wav_fn = os.path.basename(wav_path)
#
#     try:
#         if os.path.exists(spectrogram_path):
#             n_frames = np.load(spectrogram_path).shape[1]
#         else:
#             # Load the audio to a numpy array:
#             wav = audio.load_wav(wav_path)
#
#             # Compute the linear-scale spectrogram from the wav:
#             spectrogram = audio.spectrogram(wav).astype(np.float32)
#             n_frames = spectrogram.shape[1]
#
#             # Write the spectrograms to disk:
#             np.save(spectrogram_path, spectrogram.T, allow_pickle=False)
#
#             if not os.path.exists(mel_path):
#                 # Compute a mel-scale spectrogram from the wav:
#                 mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
#
#                 # Write the spectrograms to disk:
#                 np.save(mel_path, mel_spectrogram.T, allow_pickle=False)
#
#         # Return a tuple describing this training example:
#         return (wav_fn, spectrogram_filename, mel_filename, n_frames, text)
#     except:
#         return None
