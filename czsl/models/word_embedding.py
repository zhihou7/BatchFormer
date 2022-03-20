import torch
import numpy as np
from flags import DATA_FOLDER
def load_word_embeddings(emb_type, vocab):
    if emb_type == 'glove':
        embeds = load_glove_embeddings(vocab)
    elif emb_type == 'fasttext':
        embeds = load_fasttext_embeddings(vocab)
    elif emb_type == 'word2vec':
        embeds = load_word2vec_embeddings(vocab)
    elif emb_type == 'ft+w2v':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim = 1)
        print('Combined embeddings are ',embeds.shape)
    elif emb_type == 'ft+gl':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_glove_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim = 1)
        print('Combined embeddings are ',embeds.shape)
    elif emb_type == 'ft+ft':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds1], dim = 1)
        print('Combined embeddings are ',embeds.shape)
    elif emb_type == 'gl+w2v':
        embeds1 = load_glove_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim = 1)
        print('Combined embeddings are ',embeds.shape)
    elif emb_type == 'ft+w2v+gl':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds3 = load_glove_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2, embeds3], dim = 1)
        print('Combined embeddings are ',embeds.shape)
    else:
        raise ValueError('Invalid embedding')
    return embeds

def load_fasttext_embeddings(vocab):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    import fasttext.util
    ft = fasttext.load_model(DATA_FOLDER+'/fast/cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)

    embeds = torch.Tensor(np.stack(embeds))
    print('Fasttext Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds

def load_word2vec_embeddings(vocab):
    # vocab = [v.lower() for v in vocab]

    from gensim import models
    model = models.KeyedVectors.load_word2vec_format(
        DATA_FOLDER+'/w2v/GoogleNews-vectors-negative300.bin', binary=True)

    custom_map = {
        'Faux.Fur': 'fake_fur',
        'Faux.Leather': 'fake_leather',
        'Full.grain.leather': 'thick_leather',
        'Hair.Calf': 'hair_leather',
        'Patent.Leather': 'shiny_leather',
        'Boots.Ankle': 'ankle_boots',
        'Boots.Knee.High': 'knee_high_boots',
        'Boots.Mid-Calf': 'midcalf_boots',
        'Shoes.Boat.Shoes': 'boat_shoes',
        'Shoes.Clogs.and.Mules': 'clogs_shoes',
        'Shoes.Flats': 'flats_shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k and k not in model:
            ks = k.split('_')
            emb = np.stack([model[it] for it in ks]).mean(axis=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    print('Word2Vec Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds

def load_glove_embeddings(vocab):
    '''
    Inputs
        emb_file: Text file with word embedding pairs e.g. Glove, Processed in lower case.
        vocab: List of words
    Returns
        Embedding Matrix
    '''
    vocab = [v.lower() for v in vocab]
    emb_file = DATA_FOLDER+'/glove/glove.6B.300d.txt'
    model = {}  # populating a dictionary of word and embeddings
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')  # Word-embedding
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        model[line[0]] = wvec

    # Adding some vectors for UT Zappos
    custom_map = {
        'faux.fur': 'fake_fur',
        'faux.leather': 'fake_leather',
        'full.grain.leather': 'thick_leather',
        'hair.calf': 'hair_leather',
        'patent.leather': 'shiny_leather',
        'boots.ankle': 'ankle_boots',
        'boots.knee.high': 'knee_high_boots',
        'boots.mid-calf': 'midcalf_boots',
        'shoes.boat.shoes': 'boat_shoes',
        'shoes.clogs.and.mules': 'clogs_shoes',
        'shoes.flats': 'flats_shoes',
        'shoes.heels': 'heels',
        'shoes.loafers': 'loafers',
        'shoes.oxfords': 'oxford_shoes',
        'shoes.sneakers.and.athletic.shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower',
        'nubuck' : 'grainy_leather',
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k:
            ks = k.split('_')
            emb = torch.stack([model[it] for it in ks]).mean(dim=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.stack(embeds)
    print('Glove Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds