import argparse
import re
import os
import numpy as np


def get_states(data, pos, masking_matrix, idx_matrix, state_labels, length_matrix, offset_matrix, shifting_matrix, modified_length):
    cleaned = {}

    byte_length = np.ceil(length_matrix.astype('float16') / 8).astype('uint8')

    masked = np.bitwise_and(masking_matrix, data[idx_matrix])
    shifted = np.right_shift(masked, offset_matrix)
    left_shifted = np.left_shift(shifted, shifting_matrix)
    
    state_packet = []

    byte_position = 0
    for item in modified_length:
        payload = 0

        for i in range(item):
            payload |= left_shifted[byte_position + i]

        state_packet.append(payload)

        byte_position += item

    return state_packet


def generate_idx_matrix(pos):
    state_labels = []
    idx_matrix = np.array([], dtype='uint8')
    masking_matrix = np.array([], dtype='uint8') 
    length_matrix = np.array([], dtype='uint8') 
    offset_matrix = np.array([], dtype='uint8')
    shifting_matrix = np.array([], dtype='uint8')
    modified_length = []

    for key in pos:
        bit_location = pos[key]['bitLocation']
        byte_location = pos[key]['byteLocation']
        length = pos[key]['length']

        offset = (length + bit_location) // 8

        # Only if the range is in the same byte
        if offset == 0:
            offset += 1

        # Add one more byte if it exceeds the boundaries
        if (length + bit_location) >= 8 and (length + bit_location) % 8 > 0:
            offset += 1

        byte_range = np.arange(byte_location, byte_location + offset)

        state_labels.append(key)
        idx_matrix = np.hstack([
            idx_matrix,
            byte_range
        ])

        offset_vector, shifting_vector, running_mask = generate_mask(pos[key])

        modified_length.append(offset)

        masking_matrix = np.hstack([masking_matrix, running_mask])
        length_matrix = np.hstack([length_matrix, length])
        offset_matrix = np.hstack([offset_matrix, offset_vector])
        shifting_matrix = np.hstack([shifting_matrix, shifting_vector])

    return state_labels, idx_matrix, masking_matrix, length_matrix, offset_matrix, shifting_matrix, modified_length


def generate_mask(state):
    """
    Generate a binary mask for the state vector
    """

    mask = []

    offset = state['bitLocation']
    running_mask = 1
    bit_count = offset + 1

    offset_vector = []
    shifting_vector = []

    for i in range(state['length']):
        if bit_count == 8:
            running_mask = running_mask << offset
            mask.append(running_mask)

            offset_vector.append(offset)
            shifting_vector.append(i - bit_count + 1)

            offset = 0
            running_mask = 1
            bit_count = 1

        elif i == state['length'] - 1:
            offset_vector.append(offset)
            mask.append(running_mask << offset)
            shifting_vector.append(i - bit_count + 1)

        else:
            running_mask = (running_mask << 1) | 1
            bit_count += 1

    offset_vector = np.array(offset_vector)
    shifting_vector = np.array(shifting_vector) + offset_vector
    mask = np.array(mask)

    return offset_vector, shifting_vector, mask


def lini_read_bcidat(filename):
    samples = 0
    filesize = os.path.getsize(filename)

    with open(filename, 'rb') as f:
        header_line = f.readline()
        header = {}

        for item in re.findall(r'[\w]+=\s[\w.]+', header_line.decode('utf-8')):
            k, v = item.split()

            try:
                v = int(v)
            except:
                pass

            header[k[:-1]] = v

        f.seek(0)
        chunk = f.read(header['HeaderLen']).decode('utf-8')

        # State vectors
        svs = re.search(r'(?<=\[ State Vector Definition \][\S\s])[\S\s]*(?=\[\sParameter Definition \])', chunk).group()
        svs = svs.rstrip('\r\n').lstrip('\r\n')

        # Rest of header file
        stripped = re.findall(r'[\w]+=\s.+(?=\/\/)', chunk)
        for item in stripped:
            k, v = item.split('=')

            if k not in header:
                header[k] = v.lstrip().rstrip()

        channel_gain = np.array(
            list(
                map(
                    float,
                    header['SourceChGain'].split()[2:header['SourceCh'] + 2]
                )
            )
        )

        channel_gain = np.diag(channel_gain)

        header['DataLength'] = 4

        if header['DataFormat'] == 'int16':
            header['DataLength'] = 2

        samples = (filesize - header['HeaderLen']) // (header['DataLength'] * header['SourceCh'] + header['StatevectorLen'])
        signal = np.zeros((samples, header['SourceCh']))

        internal_svs = {}
        for state in svs.split('\r\n'):
            values = state.split()
            fmts = list(map(int, values[1:]))

            name = values[0]
            bits = {
                'length': fmts[0],
                'value': fmts[1],
                'byteLocation': fmts[2],
                'bitLocation': fmts[3],
            }

            internal_svs[name] = bits

        fs = int(header['SamplingRate'].split()[0])
        state_vectors = np.zeros([samples, len(internal_svs)])

        state_labels, idx_matrix, masking_matrix, length_matrix, offset_matrix, shifting_matrix, modified_length = generate_idx_matrix(internal_svs)

        f.seek(header['HeaderLen'])
        for i in range(samples):
            # Get data for all channels
            # TODO: auto change dtype

            data = np.frombuffer(f.read(header['DataLength'] * header['SourceCh']), dtype='<f4').reshape(1, -1)
            data = np.matmul(channel_gain, data.T).T

            s_ve = np.frombuffer(f.read(header['StatevectorLen']), dtype=np.uint8)

            states = get_states(s_ve, internal_svs, masking_matrix, idx_matrix, state_labels, length_matrix, offset_matrix, shifting_matrix, modified_length)
            state_vectors[i, :] = states

            signal[i, :] = data

        return (signal, state_vectors, state_labels, fs)
