import re


class Seq:
    def __init__(self, input_file):
        """
        Load a FLIR SEQ file. Currently, this must be a SEQ
        file containing FFF files. The resulting object can
        be indexed as a normal array and will return the
        """
        with open(input_file, 'rb') as seq_file:
            self.seq_blob = seq_file.read()

        self._fff_it = self._get_fff_iterator(self.seq_blob)

        self.pos = []
        prev_pos = 0

        # Iterate through sequence to get frame offsets
        for match in self._fff_it:
            index = match.start()
            chunksize = index - prev_pos
            self.pos.append((index, chunksize))
            prev_pos = index

        # Fix up the first chunk size
        if len(self.pos) > 1:
            self.pos[0] = (0, self.pos[1][1])
        elif len(self.pos) == 1:
            self.pos[0] = (0, len(self.seq_blob))

    @staticmethod
    def _get_fff_iterator(seq_blob):
        """
        Internal function which returns an iterator containing the
        indices of the files in the SEQ. Probably this should be
        converted to something a bit more intelligent which
        actually identifies the size of the records in the file.
        """
        magic_pattern_fff = "\x46\x46\x46\x00".encode()

        valid = re.compile(magic_pattern_fff)
        return valid.finditer(seq_blob)

    def __len__(self):
        """
        Returns the length of the sequence
        """
        return len(self.pos)

    def __getitem__(self, index):
        """
        Return a byte string containing the FFF image in the sequence
        """

        offset, chunksize = self.pos[index]
        chunk = self.seq_blob[offset:offset + chunksize]

        return chunk
