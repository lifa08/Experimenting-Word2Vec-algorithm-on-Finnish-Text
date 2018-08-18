from voikko.libvoikko import Voikko
import os
from collections import Counter
import random
import urllib
import tarfile
import numpy as np
import pickle as pkl

def safe_mkdir(path):
    '''Create a directory if there isn't one already.'''
    try:
        os.makedirs(path)
    except OSError:
        pass

def unzip_and_remove(zipped_file, unzip_dir):
    print('unzipping file...')
    tar = tarfile.open(zipped_file, 'r')
    tar.extractall(path=unzip_dir)
    tar.close()
    os.remove(zipped_file)

def download_data_file(download_url,
                       data_dir,
                       local_dest, 
                       expected_byte):
    """ 
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    Check if the downloaded file has the same number of bytes as expected_byte.
    Unzip the file and remove the zip file
    """
    unzip_name = local_dest[:-4]

    if os.path.exists(unzip_name):
        print('file already exists')
        return unzip_name
    elif os.path.exists(local_dest):
        print('file already exists but unzipped')
        unzip_and_remove(local_dest, data_dir)
        return unzip_name
    else:
        safe_mkdir(data_dir)

        print('Downloading...')
        _, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)

        if file_stat.st_size == expected_byte:
            print('Successfully downloaded')
        else:
            print('The downloaded file has unexpected number of bytes')
            return

        unzip_and_remove(local_dest, data_dir)
        return unzip_name

def read_data(file_path):
    ''' Read data into a list of words and store the words into a file
    if the relevant word file does not exist'''

    if os.path.exists(file_path + '_words'):
        print('reading from word file...')
        with open(file_path + '_words', 'r') as f:
            words = f.read().split('\n')
            return words

    print('reading from data file...')
    v = Voikko("fi")

    with open(file_path) as f:
        words = [word.tokenText.lower() for word in v.tokens(f.read())
                 if word.tokenType==1 or word.tokenType==2]
        v.terminate()

        file = open(file_path + '_words', 'w')
        file.write('\n'.join(words))
        file.close()

        return words

def build_vocab(words, vocab_size, vocab_dir, vocab_file_path):
    '''Build vocabulary of vocab_size most frequent words and write it to vocab_file_path.
        words: a list of words.
    '''
    print("building vocabulary...")

    dictionary = dict()
    index = 0

    if words == None:
        with open(vocab_file_path, 'r') as f:
            words = f.read().split('\n')
            for word in words:
                dictionary[word] = index
                index += 1
    else:
        safe_mkdir(vocab_dir)
        file = open(vocab_file_path, 'w')
        count = [('UNK', -1)]
        count.extend(Counter(words).most_common(vocab_size - 1))

        for word, _ in count:
            dictionary[word] = index
            index += 1
            file.write(word + '\n')
        file.close()

    return dictionary

def sentence_to_index(index_file, file_path, dictionary):
    '''Read sentences from file and replace them with
    their corresponding word indices in the dictionary'''

    print("converting sentences to indices...")
    v = Voikko("fi")

    index_f = open(index_file, 'wb')
    with open(file_path) as f:
        index_sentences = []
        for sentence in f:
            words = [word.tokenText.lower() for word in v.tokens(sentence)
                 if word.tokenType==1 or word.tokenType==2]

            index_words = [dictionary[word] if word in dictionary else 0 for word in words]
            index_sentences.append(index_words)
        v.terminate()

        # save sentence indices into a index_file
        pkl.dump(index_sentences, index_f, -1)
        index_f.close()
        
        return index_sentences

def generate_sample(index_words, context_window_size):
    '''Form training pairs according to the skip-gram model.'''
    for sentence_words in index_words:
        for index, center in enumerate(sentence_words):
            context = random.randint(1, context_window_size)

            # get a random number of targets before the center word
            for target in sentence_words[max(0, index - context): index]:
                yield center, target

            # get a random number of targets after the center word
            for target in sentence_words[index + 1: index + context + 1]:
                yield center, target

def batch_gen(download_url, data_dir, data_file_path, expected_byte,
              vocab_dir, vocab_file_path, vocab_size, index_file,
              batch_size, skip_window, lemmatize):
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            sentence_indices = pkl.load(f)

    else:
        if os.path.exists(vocab_file_path):
            dictionary =  build_vocab(None, 0, vocab_dir, vocab_file_path)
        else:
            data_file_path = download_data_file(download_url, data_dir, data_file_path, expected_byte)
            if lemmatize:
                data_file_path = lemmatize_file(data_file_path)
            words = read_data(data_file_path)
            dictionary = build_vocab(words, vocab_size, vocab_dir, vocab_file_path)
            del words

        sentence_indices = sentence_to_index(index_file, data_file_path, dictionary)

    single_gen = generate_sample(sentence_indices, skip_window)

    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch

def most_common_words(vocab_file_path, visual_dir, num_visualize):
    """ create a list of num_visualize most frequent words to visualize on TensorBoard.
    saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(vocab_file_path, 'r').readlines()[:num_visualize]
    words = [word for word in words]

    safe_mkdir(visual_dir)
    file = open(os.path.join(visual_dir, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()

def lemmatize_file(filename):
    print('lemmatizing ' + filename)

    v = Voikko("fi")
    lemmatized_filename = filename + '_lemmatized'
    lemmatized_file = open(lemmatized_filename, 'w') 

    with open(filename, 'r') as f:
        for sentence in f:
            sent_toks = v.tokens(sentence)

            words_baseform = []
            for word in sent_toks:
                if word.tokenType == 1:
                    word_analyzed = v.analyze(word.tokenText)
                    if len(word_analyzed) > 0:
                        words_baseform.append(word_analyzed[0].get('BASEFORM'))
                    else:
                        words_baseform.append(word.tokenText)
                else:
                    words_baseform.append(word.tokenText)

            sent_baseform = ''.join(words_baseform)
            lemmatized_file.write(sent_baseform)

    lemmatized_file.close()
    v.terminate()
    return lemmatized_filename
