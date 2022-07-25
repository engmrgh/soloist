from itertools import chain


def history_encoder(tokenizer, history):
    """ Encode the history as a sequence of input ids. """
    system_id, user_id = tokenizer.convert_tokens_to_ids(['<system>', '<user>'])
    speaker_id_dict = {'user': [user_id], 'system': [system_id]}

    history_encoded = list()
    for item in history:
        speaker, utterance = item.split(':', maxsplit=1)

        speaker_ids = speaker_id_dict[speaker.strip().lower()]
        colon_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(':'))
        utterance_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utterance))

        history_encoded.append(speaker_ids + colon_ids + utterance_ids)

    return history_encoded


def belief_encoder(tokenizer, belief):
    _, info = belief.split(':', maxsplit=1)
    belief_ids = tokenizer.convert_tokens_to_ids(["<belief>"])
    colon_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(':'))
    info_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(info))
    belief_encoded = belief_ids + colon_ids + info_ids
    return belief_encoded


def kb_encoder(tokenizer, kb):
    _, info = kb.split(':', maxsplit=1)
    kb_ids = tokenizer.convert_tokens_to_ids(["<db>"])
    colon_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(':'))
    info_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(info))
    kb_encoded = kb_ids + colon_ids + info_ids
    return kb_encoded


def reply_encoder(tokenizer, reply):
    _, response = reply.split(':', maxsplit=1)
    reply_encoded = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))
    return reply_encoded


def segments_encoder(tokenizer, history, belief, kb, reply):
    out = 4*[None]
    out[0] = history_encoder(tokenizer, history) if history else []
    out[1] = belief_encoder(tokenizer, belief) if belief else []
    out[2] = kb_encoder(tokenizer, kb) if kb else []
    out[3] = reply_encoder(tokenizer, reply) if reply else []
    return tuple(out)


def create_input_ids(sequence):
    return list(chain(*sequence))


def create_token_type_ids(tokenizer, sequence):
    system_id, user_id = tokenizer.convert_tokens_to_ids(['<system>', '<user>'])
    get_other_speaker_id = {system_id: user_id, user_id: system_id}

    if len(sequence) == 2:
        history, belief = sequence
        kb, reply = [], []
    elif len(sequence) == 4:
        history, belief, kb, reply = sequence
    else:
        raise('Invalid sequence length')

    token_type_ids = list()
    ptr = 0
    while ptr < len(history):
        speaker_id = history[ptr]
        try:
            new_ptr = history.index(get_other_speaker_id[speaker_id], ptr+1)
        except ValueError:
            new_ptr = len(history)
        token_type_ids += [speaker_id] * (new_ptr - ptr)
        ptr = new_ptr

    token_type_ids += [system_id] * (len(belief) + len(kb) + len(reply))
    return token_type_ids


def create_lm_labels(sequence, fake=False):
    if fake:
        lm_labels = [-100] * len(list(chain(*sequence)))
    else:
        lm_labels = list()
        for i, segment in enumerate(sequence):
            segment_lm_labels = [-100] * len(segment) if i%2==0 else segment
            lm_labels.extend(segment_lm_labels)

    return lm_labels
